import os
import copy
import random
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import boto3
from botocore.exceptions import ClientError
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets import ImageFolder
from sklearn.metrics import recall_score, f1_score
from sklearn.model_selection import train_test_split

import time

# Custom Modüller (Repo'dan gelen)
from Model.resnet import ResNet
from options import args_parser
from Dataset.dataset import (classify_label, Indices2Dataset_labeled,
                              Indices2Dataset_unlabeled_fixmatch, partition_train)
from Dataset.sample_dirichlet import clients_indices, clients_indices_homo

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_exp_name(args):
    """STFL eklentili yeni deney ismi"""
    return (f"{args.dataset}_a{args.alpha}_STFL-{args.aggregation_method}_"
            f"L{args.num_labeled}_C{args.num_clients}-ON{args.num_online_clients}_"
            f"E{args.local_epochs}_T{args.threshold}")
  
def save_checkpoint(round_num, model_state, metrics_history, local_ckpt_dir, args, filename='checkpoint.pt', backup_every=3):
    folder_name = get_exp_name(args)
    os.makedirs(local_ckpt_dir, exist_ok=True)
    state = {
        'round': round_num,
        'model_state_dict': model_state,
        'metrics_history': metrics_history,
        'args': args 
    }
    
    local_path = os.path.join(local_ckpt_dir, filename)
    torch.save(state, local_path)
    
    # S3 Yedekleme
    if round_num % backup_every == 0 or round_num == args.num_rounds:
        s3 = boto3.client('s3')
        s3_ckpt_path = f"checkpoints/{folder_name}/{filename}"
        s3_csv_path  = f"results/{folder_name}/STFL_{args.aggregation_method}_alpha={args.alpha}.csv"
        
        try:
            s3.upload_file(local_path, args.s3_bucket, s3_ckpt_path)
            local_csv_path = f'./results/{args.dataset}/STFL_{args.aggregation_method}_alpha={args.alpha}.csv'
            if os.path.exists(local_csv_path):
                s3.upload_file(local_csv_path, args.s3_bucket, s3_csv_path)
            print(f"[S3-SYNC] Round {round_num} yedeklendi (STFL aktif).")
        except Exception as e:
            print(f"[WARNING] S3 Backup hatası: {e}")

def load_checkpoint(model, local_ckpt_dir, args, filename='checkpoint.pt'):
    s3 = boto3.client('s3')
    folder_name = get_exp_name(args)
    local_path = os.path.join(local_ckpt_dir, filename)
    s3_path = f"checkpoints/{folder_name}/{filename}"

    if not os.path.exists(local_path):
        try:
            print(f"Downloading checkpoint from S3: s3://{args.s3_bucket}/{s3_path}")
            os.makedirs(local_ckpt_dir, exist_ok=True)
            s3.download_file(args.s3_bucket, s3_path, local_path)
        except ClientError:
            print("No checkpoint found. Starting STFL training from scratch.")
            return 1, {'acc': [], 'acsa': [], 'f1': []}
    
    print(f"[CKPT] Loading checkpoint: {local_path}")
    try:
        ckpt = torch.load(local_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        return ckpt['round'] + 1, ckpt['metrics_history']
    except Exception as e:
        print(f"[CKPT] Load error: {e}  →  Starting from scratch.")
        return 1, {'acc': [], 'acsa': [], 'f1': []}

# ══════════════════════════════════════════════════════════════
#  SHAPLEY  (CSSV)
# ══════════════════════════════════════════════════════════════
def compute_cssv(args, local_models_params, initial_global_params):
    num_clients = len(local_models_params)
    num_classes = args.num_classes
  
    if num_clients == 0: return np.array([])

    weight_layer = 'classifier.weight'
    bias_layer   = 'classifier.bias'
    client_updates = []
    for local_params in local_models_params:
        update = {}
        update[weight_layer] = local_params[weight_layer] - initial_global_params[weight_layer].to(local_params[weight_layer].device)
        update[bias_layer] = local_params[bias_layer] - initial_global_params[bias_layer].to(local_params[bias_layer].device)
        client_updates.append(update)
      
    shapley_values = np.zeros((num_clients, num_classes))
    num_samples = getattr(args, 'shapley_samples', 10)

    for _ in range(num_samples):
        permutation = np.random.permutation(num_clients)
        for i, client_idx in enumerate(permutation):
            coalition_indices = permutation[:i]
            coalition_plus_indices = permutation[:i+1]

            for c in range(num_classes):
                curr_w_c = torch.cat([client_updates[client_idx][weight_layer][c].view(-1), client_updates[client_idx][bias_layer][c].view(-1)])
                curr_w_norm = F.normalize(curr_w_c.unsqueeze(0), p=2) if torch.norm(curr_w_c) > 0 else curr_w_c.unsqueeze(0)

                sim_s = 0.0
                if len(coalition_indices) > 0:
                    tw_c = sum(client_updates[co_idx][weight_layer][c] for co_idx in coalition_indices) / len(coalition_indices)
                    tb_c = sum(client_updates[co_idx][bias_layer][c] for co_idx in coalition_indices) / len(coalition_indices)
                    agg_c = torch.cat([tw_c.view(-1), tb_c.view(-1)])
                    if torch.norm(agg_c) > 0 and torch.norm(curr_w_c) > 0:
                        sim_s = F.cosine_similarity(curr_w_norm, F.normalize(agg_c.unsqueeze(0), p=2)).item()

                tw2_c = sum(client_updates[co_idx][weight_layer][c] for co_idx in coalition_plus_indices) / len(coalition_plus_indices)
                tb2_c = sum(client_updates[co_idx][bias_layer][c] for co_idx in coalition_plus_indices) / len(coalition_plus_indices)
                agg2_c = torch.cat([tw2_c.view(-1), tb2_c.view(-1)])
                
                sim_si = 0.0
                if torch.norm(agg2_c) > 0 and torch.norm(curr_w_c) > 0:
                    sim_si = F.cosine_similarity(curr_w_norm, F.normalize(agg2_c.unsqueeze(0), p=2)).item()

                shapley_values[client_idx, c] += (sim_si - sim_s)

    if num_samples > 0: shapley_values /= num_samples
    shapley_values = np.maximum(shapley_values, 0)
    
    for c in range(num_classes):
        col_sum = np.sum(shapley_values[:, c])
        if col_sum > 0: shapley_values[:, c] /= col_sum
        else: shapley_values[:, c] = 1.0 / num_clients

    return shapley_values

# ══════════════════════════════════════════════════════════════
#  GLOBAL MODEL
# ══════════════════════════════════════════════════════════════
def patch_resnet_for_ham(model):
    model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    return model

class Global(object):
    def __init__(self, args):
        self.model = ResNet(resnet_size=8, scaling=4, save_activations=False, 
                            group_norm_num_groups=None, freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
        if args.dataset == 'HAM10000':
            patch_resnet_for_ham(self.model)
        self.model.cuda(args.gpu_id)
        self.num_classes = args.num_classes

    def initialize_for_model_fusion(self, args, list_dicts_local_params, list_nums_local_data, initial_global_params):
        fused_params = copy.deepcopy(list_dicts_local_params[0])
        num_clients = len(list_dicts_local_params)
        total_data = sum(list_nums_local_data)
        
        if args.aggregation_method == 'ShapFed':
            cssv_weights = compute_cssv(args, list_dicts_local_params, initial_global_params)
            client_backbone_weights = np.mean(cssv_weights, axis=1)
            if np.sum(client_backbone_weights) > 0:
                client_backbone_weights /= np.sum(client_backbone_weights)
            else:
                client_backbone_weights = np.ones(num_clients) / num_clients
        else:
            client_backbone_weights = [n / total_data for n in list_nums_local_data]

        for name_param in list_dicts_local_params[0]:
            if args.aggregation_method != 'ShapFed':
                fused_tensor = sum(list_dicts_local_params[i][name_param] * list_nums_local_data[i] for i in range(num_clients)) / total_data
            else:
                if name_param == 'classifier.weight' or name_param == 'classifier.bias':
                    fused_tensor = torch.zeros_like(list_dicts_local_params[0][name_param], dtype=torch.float32)
                    for c in range(args.num_classes):
                        for i in range(num_clients):
                            fused_tensor[c] += list_dicts_local_params[i][name_param][c] * cssv_weights[i, c]
                else:
                    fused_tensor = sum(list_dicts_local_params[i][name_param] * client_backbone_weights[i] for i in range(num_clients))

            if list_dicts_local_params[0][name_param].dtype == torch.int64:
                fused_params[name_param] = fused_tensor.to(torch.int64)
            else:
                fused_params[name_param] = fused_tensor
                
        return fused_params

    def fedavg_eval(self, fedavg_params, data_test, batch_size_test, args):
        self.model.load_state_dict(fedavg_params)
        self.model.eval()
        all_labels, all_predicts = [], []
        num_corrects = 0
        with torch.no_grad():
            for images, labels in DataLoader(data_test, batch_size_test):
                images, labels = images.cuda(args.gpu_id), labels.cuda(args.gpu_id)
                _, outputs = self.model(images)
                _, predicts = torch.max(outputs, -1)
                num_corrects += torch.sum(torch.eq(predicts.cpu(), labels.cpu())).item()
                all_labels.extend(labels.cpu().numpy())
                all_predicts.extend(predicts.cpu().numpy())
        
        accuracy = num_corrects / len(data_test)
        acsa = recall_score(all_labels, all_predicts, average='macro', zero_division=0)
        macro_f1 = f1_score(all_labels, all_predicts, average='macro', zero_division=0)
        
        torch.cuda.empty_cache()
        return accuracy, acsa, macro_f1

    def download_params(self):
        return self.model.state_dict()

# ══════════════════════════════════════════════════════════════
#  LOCAL MODEL (STFL EKLENTİLİ)
# ══════════════════════════════════════════════════════════════
class Local(object):
    def __init__(self, args):
        self.local_model = ResNet(resnet_size=8, scaling=4, save_activations=False, 
                                  group_norm_num_groups=None, freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
        self.local_G = ResNet(resnet_size=8, scaling=4, save_activations=False, 
                              group_norm_num_groups=None, freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
        if args.dataset == 'HAM10000':
            patch_resnet_for_ham(self.local_model)
            patch_resnet_for_ham(self.local_G)
            
        self.local_model.cuda(args.gpu_id)
        self.local_G.cuda(args.gpu_id)
        self.criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu_id)
        self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=args.lr_local_training, momentum=0.9, weight_decay=1e-4)

        # STFL: Sınıf olasılıklarının hareketli ortalamasını tutacağımız tensor
        self.class_probs_ema = torch.ones(args.num_classes).cuda(args.gpu_id) / args.num_classes

    def fixmatch_train(self, args, data_client_labeled, data_client_unlabeled, global_params, r):
        self.labeled_trainloader = DataLoader(dataset=data_client_labeled, sampler=RandomSampler(data_client_labeled),
                                              batch_size=args.batch_size_local_labeled_fixmatch, drop_last=True, num_workers=2, pin_memory=True)
        self.unlabeled_trainloader = DataLoader(dataset=data_client_unlabeled, sampler=RandomSampler(data_client_unlabeled),
                                                batch_size=args.batch_size_local_labeled_fixmatch * args.mu, drop_last=True, num_workers=2, pin_memory=True)
        
        self.local_model.load_state_dict(global_params)
        self.local_model.train()
        self.local_G.load_state_dict(global_params)
        self.local_G.eval()

        for local_epoch in range(args.local_epochs):
            labeled_iter   = iter(self.labeled_trainloader)
            unlabeled_iter = iter(self.unlabeled_trainloader)
            local_iter = int(len(data_client_unlabeled) / args.batch_size_local_labeled_fixmatch)

            for epoch in range(local_iter):
                try: inputs_x, targets_x = next(labeled_iter)
                except StopIteration: labeled_iter = iter(self.labeled_trainloader); inputs_x, targets_x = next(labeled_iter)

                try: inputs_u_w, inputs_u_s, targets_u_gt = next(unlabeled_iter)
                except StopIteration: unlabeled_iter = iter(self.unlabeled_trainloader); inputs_u_w, inputs_u_s, targets_u_gt = next(unlabeled_iter)

                inputs_x   = inputs_x.cuda(args.gpu_id)
                inputs_u_w = inputs_u_w.cuda(args.gpu_id)
                inputs_u_s = inputs_u_s.cuda(args.gpu_id)
                targets_x  = targets_x.cuda(args.gpu_id)
                batch_size = inputs_x.shape[0]

                inputs = self.interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1).cuda(args.gpu_id)

                _, logits = self.local_model(inputs)
                logits    = self.de_interleave(logits, 2 * args.mu + 1)
                logits_x  = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                del logits

                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

                _, logits_u_w_global = self.local_G(inputs_u_w)
                pseudo_label_global  = torch.softmax(logits_u_w_global.detach() / args.T, dim=-1)
                max_probs_global, targets_u_global = torch.max(pseudo_label_global, dim=-1)

                pseudo_label_local = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
                max_probs_local, targets_u_local = torch.max(pseudo_label_local, dim=-1)

                # --- YENİ EKLENEN KISIM: STFL DINAMIK EŞİK ---
                # EMA (Hareketli ortalama) güncellemesi
                self.class_probs_ema = self.class_probs_ema * 0.999 + pseudo_label_local.mean(dim=0) * 0.001
                
                # Sınıf bazlı dinamik eşikleri hesapla: p_c / max(p) * threshold
                max_ema = self.class_probs_ema.max()
                dynamic_thresholds = args.threshold * (self.class_probs_ema / max_ema)
                
                # Bu batch'teki tahminlerin dinamik eşiklerini al
                batch_thresholds_local = dynamic_thresholds[targets_u_local]
                batch_thresholds_global = dynamic_thresholds[targets_u_global]

                # Sabit args.threshold YERİNE dinamik eşikleri kullanıyoruz!
                mask_local  = max_probs_local.ge(batch_thresholds_local).float()
                mask_global = max_probs_global.ge(batch_thresholds_global).float()
                # ---------------------------------------------

                targets_u_local_one_hot  = F.one_hot(targets_u_local,  args.num_classes).float()
                targets_u_global_one_hot = F.one_hot(targets_u_global, args.num_classes).float()

                delta_c = torch.clamp(torch.abs(max_probs_local - max_probs_global) + 1e-6, min=1e-6, max=1.0)
                kappa = torch.log(torch.tensor(2.0)) / 0.05
                lambda_dynamic = torch.clamp(torch.exp(-kappa * delta_c), min=1e-6, max=1.0)

                final_targets_u = torch.where(
                    mask_local.unsqueeze(1).bool(),
                    lambda_dynamic.unsqueeze(1) * targets_u_local_one_hot + (1 - lambda_dynamic).unsqueeze(1) * targets_u_global_one_hot,
                    targets_u_global_one_hot
                )
                mask_valid = torch.max(mask_local, mask_global)

                logits_u_s_probs = torch.softmax(logits_u_s, dim=-1) + 1e-10
                final_targets_u  = final_targets_u + 1e-10

                Lu   = (F.kl_div(logits_u_s_probs.log(), final_targets_u, reduction='none').sum(-1) * mask_valid).mean()
                loss = Lx + args.lambda_u * Lu

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        final_state = {k: v.cpu() for k, v in self.local_model.state_dict().items()}
        self.optimizer.zero_grad(set_to_none=True)
        return final_state

    def interleave(self, x, size):
        s = list(x.shape)
        return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

    def de_interleave(self, x, size):
        s = list(x.shape)
        return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

# ══════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════
class _SubsetImageFolder(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices, transform=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
        max_len = len(base_dataset)
        self.valid_indices = [i for i in indices if i < max_len]
        self.targets = [base_dataset.targets[i] for i in self.valid_indices]

    def __len__(self): return len(self.valid_indices)

    def __getitem__(self, idx):
        try:
            real_idx = self.valid_indices[idx]
            img, label = self.base_dataset[real_idx]
            if self.transform: img = self.transform(img)
            return img, label
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self.valid_indices))

def main_loop(alpha):
    args = args_parser()
    args.alpha = alpha
    args.s3_bucket = 'sage-ham10k-eda'
    
    exp_name = get_exp_name(args)
    local_ckpt_dir = os.path.join(args.checkpoint_dir, exp_name)

    log_dir  = f'./results/{args.dataset}/logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{exp_name}.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file, filemode='a')

    if args.dataset == 'HAM10000':
        args.num_classes = 7
        args.num_rounds  = 300
        ham_mean, ham_std = [0.763, 0.545, 0.570], [0.140, 0.152, 0.169]

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=ham_mean, std=ham_std),
        ])

        full_dataset = ImageFolder(root=args.path_ham10000, transform=None)
        all_indices = list(range(len(full_dataset)))
        all_targets = [full_dataset.targets[i] for i in all_indices]

        train_indices, test_indices = train_test_split(all_indices, test_size=0.20, stratify=all_targets, random_state=args.seed)

        data_local_training = _SubsetImageFolder(full_dataset, train_indices)
        data_global_test    = _SubsetImageFolder(ImageFolder(root=args.path_ham10000, transform=transform_test), test_indices)

        from collections import Counter
        train_cls = Counter(all_targets[i] for i in train_indices)
        cls_names = full_dataset.classes
        print(f"[HAM10000] STFL Training Active | Train Dist: { {cls_names[k]: v for k,v in sorted(train_cls.items())} }")

    os.makedirs(local_ckpt_dir, exist_ok=True)
    random_state = np.random.RandomState(args.seed)
    
    list_label2indices = classify_label(data_local_training, args.num_classes)
    ipc = args.num_labeled
    list_label2indices_labeled, list_label2indices_unlabeled = partition_train(list_label2indices, ipc)
  
    if alpha == 0:
        list_client2indices_labeled   = clients_indices_homo(list_label2indices_labeled, args.num_classes, args.num_clients)
        list_client2indices_unlabeled = clients_indices_homo(list_label2indices_unlabeled, args.num_classes, args.num_clients)
    else:
        list_client2indices_labeled   = clients_indices(list_label2indices_labeled, args.num_classes, args.num_clients, alpha, seed=0)
        list_client2indices_unlabeled = clients_indices(list_label2indices_unlabeled, args.num_classes, args.num_clients, alpha, seed=0)

    for client in range(len(list_client2indices_labeled)):
        if client < len(list_client2indices_unlabeled):
            list_client2indices_unlabeled[client].extend(list_client2indices_labeled[client])
        else:
            list_client2indices_unlabeled.append(list(list_client2indices_labeled[client]))
          
    global_model = Global(args)
    local_model  = Local(args)

    start_round, metrics_history = load_checkpoint(global_model.model, local_ckpt_dir, args)

    total_clients          = list(range(args.num_clients))
    indices2data_labeled   = Indices2Dataset_labeled(data_local_training)
    indices2data_unlabeled = Indices2Dataset_unlabeled_fixmatch(data_local_training)

    for r in tqdm(range(start_round, args.num_rounds + 1), desc='Server'):
        dict_global_params = global_model.download_params()
        online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)

        list_dicts_local_params = []
        list_nums_local_data    = []

        for client in online_clients:
            indices2data_labeled.load(list_client2indices_labeled[client])
            indices2data_unlabeled.load(list_client2indices_unlabeled[client])

            list_nums_local_data.append(len(list_client2indices_labeled[client]) + len(list_client2indices_unlabeled[client]))
          
            local_params = local_model.fixmatch_train(args, indices2data_labeled, indices2data_unlabeled, copy.deepcopy(dict_global_params), r)
            list_dicts_local_params.append(copy.deepcopy(local_params))

            del local_params
            torch.cuda.empty_cache()

        fedavg_params = global_model.initialize_for_model_fusion(args, list_dicts_local_params, list_nums_local_data, dict_global_params)
        global_model.model.load_state_dict(fedavg_params)

        acc, acsa, macro_f1 = global_model.fedavg_eval(copy.deepcopy(fedavg_params), data_global_test, args.batch_size_test, args)
        metrics_history['acc'].append(acc)
        metrics_history['acsa'].append(acsa)
        metrics_history['f1'].append(macro_f1)

        best_acc, best_acsa, best_f1 = max(metrics_history['acc']), max(metrics_history['acsa']), max(metrics_history['f1'])

        print(f"\n[Round {r:>4}/{args.num_rounds}] Acc: {acc:.4f} | ACSA: {acsa:.4f} | F1: {macro_f1:.4f} ║ Best ACSA: {best_acsa:.4f}")
        logging.info(f"Round {r} | Acc: {acc:.4f} | ACSA: {acsa:.4f} | F1: {macro_f1:.4f}")

        save_checkpoint(r, global_model.download_params(), metrics_history, local_ckpt_dir=local_ckpt_dir, args=args, backup_every=3)

        result_dir  = f'./results/{args.dataset}'
        os.makedirs(result_dir, exist_ok=True)
        result_file = f'{result_dir}/STFL_{args.aggregation_method}_alpha={alpha}.csv'
        pd.DataFrame({
            'round': list(range(1, len(metrics_history['acc']) + 1)),
            'acc': metrics_history['acc'], 'acsa': metrics_history['acsa'], 'f1': metrics_history['f1'],
        }).to_csv(result_file, index=False, encoding='utf-8')

if __name__ == '__main__':
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    torch.backends.cudnn.deterministic = True
    args = args_parser()
    main_loop(args.alpha)
