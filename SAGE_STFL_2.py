import os
import copy
import random
import logging
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import boto3
from botocore.exceptions import ClientError
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
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

# ══════════════════════════════════════════════════════════════
#  1. YENİ: FOCAL LOSS & SINIF AĞIRLIKLARI
# ══════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha  # Sınıf ağırlıkları tensorü
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        if self.reduction == "mean": return focal.mean()
        if self.reduction == "sum": return focal.sum()
        return focal

def build_loss_fn(class_counts, num_classes, device, loss_type="focal"):
    total = sum(class_counts.values()) + 1e-8
    weights = []
    for c in range(num_classes):
        count = class_counts.get(c, 1) # 0'a bölmeyi engelle
        weights.append(total / (num_classes * count))
    
    w = torch.tensor(weights, dtype=torch.float32).to(device)
    w = w / w.sum() * num_classes # Normalize
    
    if loss_type == "focal":
        return FocalLoss(alpha=w, gamma=2.0)
    return nn.CrossEntropyLoss(weight=w)

# ══════════════════════════════════════════════════════════════
#  2. YENİ: OVERSAMPLING (WEIGHTED SAMPLER)
# ══════════════════════════════════════════════════════════════
def make_weighted_sampler(dataset_labels):
    label_array = np.array(dataset_labels)
    class_counts = np.bincount(label_array)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[label_array]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=len(sample_weights),
        replacement=True
    )

def get_exp_name(args):
    return (f"{args.dataset}_a{args.alpha}_STFL-Focal-{args.aggregation_method}_"
            f"L{args.num_labeled}_C{args.num_clients}-ON{args.num_online_clients}_"
            f"E{args.local_epochs}_T{args.threshold}")
  
def save_checkpoint(round_num, model_state, metrics_history, local_ckpt_dir, args, filename='checkpoint.pt', backup_every=3):
    folder_name = get_exp_name(args)
    os.makedirs(local_ckpt_dir, exist_ok=True)
    state = {'round': round_num, 'model_state_dict': model_state, 'metrics_history': metrics_history, 'args': args}
    local_path = os.path.join(local_ckpt_dir, filename)
    torch.save(state, local_path)
    
    if round_num % backup_every == 0 or round_num == args.num_rounds:
        s3 = boto3.client('s3')
        try:
            s3.upload_file(local_path, args.s3_bucket, f"checkpoints/{folder_name}/{filename}")
            local_csv = f'./results/{args.dataset}/STFL_{args.aggregation_method}_alpha={args.alpha}.csv'
            if os.path.exists(local_csv):
                s3.upload_file(local_csv, args.s3_bucket, f"results/{folder_name}/STFL_{args.aggregation_method}.csv")
            print(f"[S3-SYNC] Round {round_num} yedeklendi.")
        except Exception as e:
            print(f"[WARNING] S3 Backup hatası: {e}")

def load_checkpoint(model, local_ckpt_dir, args, filename='checkpoint.pt'):
    s3 = boto3.client('s3')
    folder_name = get_exp_name(args)
    local_path = os.path.join(local_ckpt_dir, filename)
    s3_path = f"checkpoints/{folder_name}/{filename}"

    if not os.path.exists(local_path):
        try:
            s3.download_file(args.s3_bucket, s3_path, local_path)
        except ClientError:
            print("No checkpoint found. Starting from scratch.")
            return 1, {'acc': [], 'acsa': [], 'f1': []}
    
    print(f"[CKPT] Loading checkpoint: {local_path}")
    try:
        ckpt = torch.load(local_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        return ckpt['round'] + 1, ckpt['metrics_history']
    except Exception as e:
        print(f"[CKPT] Load error: {e}")
        return 1, {'acc': [], 'acsa': [], 'f1': []}

# ══════════════════════════════════════════════════════════════
#  SHAPLEY  (CSSV)
# ══════════════════════════════════════════════════════════════
def compute_cssv(args, local_models_params, initial_global_params):
    num_clients, num_classes = len(local_models_params), args.num_classes
    if num_clients == 0: return np.array([])

    weight_layer, bias_layer = 'classifier.weight', 'classifier.bias'
    client_updates = []
    for lp in local_models_params:
        update = {
            weight_layer: lp[weight_layer] - initial_global_params[weight_layer].to(lp[weight_layer].device),
            bias_layer: lp[bias_layer] - initial_global_params[bias_layer].to(lp[bias_layer].device)
        }
        client_updates.append(update)
      
    shapley_values = np.zeros((num_clients, num_classes))
    num_samples = getattr(args, 'shapley_samples', 10)

    for _ in range(num_samples):
        permutation = np.random.permutation(num_clients)
        for i, client_idx in enumerate(permutation):
            c_idx, cp_idx = permutation[:i], permutation[:i+1]
            for c in range(num_classes):
                curr_w_c = torch.cat([client_updates[client_idx][weight_layer][c].view(-1), client_updates[client_idx][bias_layer][c].view(-1)])
                curr_w_norm = F.normalize(curr_w_c.unsqueeze(0), p=2) if torch.norm(curr_w_c) > 0 else curr_w_c.unsqueeze(0)

                sim_s = 0.0
                if len(c_idx) > 0:
                    agg_c = torch.cat([(sum(client_updates[co][weight_layer][c] for co in c_idx) / len(c_idx)).view(-1),
                                       (sum(client_updates[co][bias_layer][c] for co in c_idx) / len(c_idx)).view(-1)])
                    if torch.norm(agg_c) > 0 and torch.norm(curr_w_c) > 0:
                        sim_s = F.cosine_similarity(curr_w_norm, F.normalize(agg_c.unsqueeze(0), p=2)).item()

                agg2_c = torch.cat([(sum(client_updates[co][weight_layer][c] for co in cp_idx) / len(cp_idx)).view(-1),
                                    (sum(client_updates[co][bias_layer][c] for co in cp_idx) / len(cp_idx)).view(-1)])
                sim_si = F.cosine_similarity(curr_w_norm, F.normalize(agg2_c.unsqueeze(0), p=2)).item() if torch.norm(agg2_c) > 0 and torch.norm(curr_w_c) > 0 else 0.0
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
        self.model = ResNet(num_classes=args.num_classes, pretrained=True)
        if args.dataset == 'HAM10000': patch_resnet_for_ham(self.model)
        self.model.cuda(args.gpu_id)
        self.num_classes = args.num_classes

    def initialize_for_model_fusion(self, args, list_dicts_local_params, list_nums_local_data, initial_global_params):
        fused_params = copy.deepcopy(list_dicts_local_params[0])
        num_clients, total_data = len(list_dicts_local_params), sum(list_nums_local_data)
        
        if args.aggregation_method == 'ShapFed':
            cssv_weights = compute_cssv(args, list_dicts_local_params, initial_global_params)
            cb_weights = np.mean(cssv_weights, axis=1)
            cb_weights = cb_weights / np.sum(cb_weights) if np.sum(cb_weights) > 0 else np.ones(num_clients) / num_clients
        else:
            cb_weights = [n / total_data for n in list_nums_local_data]

        for name in list_dicts_local_params[0]:
            if args.aggregation_method == 'ShapFed' and name in ['classifier.weight', 'classifier.bias']:
                fused_tensor = torch.zeros_like(list_dicts_local_params[0][name], dtype=torch.float32)
                for c in range(args.num_classes):
                    for i in range(num_clients):
                        fused_tensor[c] += list_dicts_local_params[i][name][c] * cssv_weights[i, c]
            else:
                fused_tensor = sum(list_dicts_local_params[i][name] * cb_weights[i] for i in range(num_clients))
            fused_params[name] = fused_tensor.to(torch.int64) if list_dicts_local_params[0][name].dtype == torch.int64 else fused_tensor
        return fused_params

    def fedavg_eval(self, fedavg_params, data_test, batch_size_test, args):
        self.model.load_state_dict(fedavg_params)
        self.model.eval()
        all_labels, all_predicts = [], []
        with torch.no_grad():
            for images, labels in DataLoader(data_test, batch_size_test):
                images, labels = images.cuda(args.gpu_id), labels.cuda(args.gpu_id)
                outputs = self.model(images)
                if isinstance(outputs, tuple): outputs = outputs[1]
                _, predicts = torch.max(outputs, -1)
                all_labels.extend(labels.cpu().numpy())
                all_predicts.extend(predicts.cpu().numpy())
        
        acc = (np.array(all_labels) == np.array(all_predicts)).mean()
        acsa = recall_score(all_labels, all_predicts, average='macro', zero_division=0)
        macro_f1 = f1_score(all_labels, all_predicts, average='macro', zero_division=0)
        
        # Sınıf bazlı detaylı rapor
        print("\n--- Per-Class Recall ---")
        classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
        for c in range(self.num_classes):
            mask = (np.array(all_labels) == c)
            if mask.sum() > 0:
                rec = ((np.array(all_predicts) == c) & mask).sum() / mask.sum()
                print(f" {classes[c]:<5}: {rec:.4f} " + "█"*int(rec*20))
        print("------------------------")
        return acc, acsa, macro_f1

    def download_params(self): return self.model.state_dict()

# ══════════════════════════════════════════════════════════════
#  LOCAL MODEL (FOCAL LOSS + STFL + OVERSAMPLING)
# ══════════════════════════════════════════════════════════════
class Local(object):
    def __init__(self, args):
        self.local_model = ResNet(num_classes=args.num_classes, pretrained=True)
        self.local_G = ResNet(num_classes=args.num_classes, pretrained=True)
        if args.dataset == 'HAM10000':
            patch_resnet_for_ham(self.local_model)
            patch_resnet_for_ham(self.local_G)
            
        self.local_model.cuda(args.gpu_id)
        self.local_G.cuda(args.gpu_id)
        self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=args.lr_local_training, momentum=0.9, weight_decay=1e-4)
        self.class_probs_ema = torch.ones(args.num_classes).cuda(args.gpu_id) / args.num_classes

    def fixmatch_train(self, args, data_client_labeled, data_client_unlabeled, global_params, r):
        
        # 1. Yerel Sınıf Dağılımını Analiz Et (Loss ve Sampler için)
        local_labels = [int(data_client_labeled[i][1]) for i in range(len(data_client_labeled))]
        class_counts = dict(zip(*np.unique(local_labels, return_counts=True)))
        
        # 2. Focal Loss Oluştur (Mevcut dağılıma göre ağırlıklandırılmış)
        self.criterion = build_loss_fn(class_counts, args.num_classes, f"cuda:{args.gpu_id}", loss_type="focal")

        # 3. Oversampling Aktif (Nadir sınıfları daha çok seçer)
        labeled_sampler = make_weighted_sampler(local_labels)
        self.labeled_trainloader = DataLoader(dataset=data_client_labeled, sampler=labeled_sampler,
                                              batch_size=args.batch_size_local_labeled_fixmatch, drop_last=True)
        
        self.unlabeled_trainloader = DataLoader(dataset=data_client_unlabeled, sampler=RandomSampler(data_client_unlabeled),
                                                batch_size=args.batch_size_local_labeled_fixmatch * args.mu, drop_last=True)
        
        self.local_model.load_state_dict(global_params)
        self.local_model.train()
        self.local_G.load_state_dict(global_params)
        self.local_G.eval()

        for local_epoch in range(args.local_epochs):
            labeled_iter = iter(self.labeled_trainloader)
            unlabeled_iter = iter(self.unlabeled_trainloader)
            local_iter = int(len(data_client_unlabeled) / args.batch_size_local_labeled_fixmatch)

            for epoch in range(local_iter):
                try: inputs_x, targets_x = next(labeled_iter)
                except StopIteration: labeled_iter = iter(self.labeled_trainloader); inputs_x, targets_x = next(labeled_iter)
                try: inputs_u_w, inputs_u_s, _ = next(unlabeled_iter)
                except StopIteration: unlabeled_iter = iter(self.unlabeled_trainloader); inputs_u_w, inputs_u_s, _ = next(unlabeled_iter)

                inputs_x, targets_x = inputs_x.cuda(args.gpu_id), targets_x.cuda(args.gpu_id)
                inputs_u_w, inputs_u_s = inputs_u_w.cuda(args.gpu_id), inputs_u_s.cuda(args.gpu_id)
                batch_size = inputs_x.shape[0]

                inputs = self.interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1)

                outputs = self.local_model(inputs)
                logits = outputs[1] if isinstance(outputs, tuple) else outputs
                logits = self.de_interleave(logits, 2 * args.mu + 1)
                
                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                
                # Standart CE yerine Focal Loss Kullanımı!
                Lx = self.criterion(logits_x, targets_x)

                # STFL & FixMatch Unlabeled Kısmı
                with torch.no_grad():
                    G_out = self.local_G(inputs_u_w)
                    logits_u_w_global = G_out[1] if isinstance(G_out, tuple) else G_out
                
                pseudo_global = torch.softmax(logits_u_w_global.detach() / args.T, dim=-1)
                max_probs_global, targets_u_global = torch.max(pseudo_global, dim=-1)

                pseudo_local = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
                max_probs_local, targets_u_local = torch.max(pseudo_local, dim=-1)

                self.class_probs_ema = self.class_probs_ema * 0.999 + pseudo_local.mean(dim=0) * 0.001
                max_ema = self.class_probs_ema.max()
                dynamic_thresholds = args.threshold * (self.class_probs_ema / max_ema)
                
                mask_local = max_probs_local.ge(dynamic_thresholds[targets_u_local]).float()
                mask_global = max_probs_global.ge(dynamic_thresholds[targets_u_global]).float()

                targets_u_local_oh = F.one_hot(targets_u_local, args.num_classes).float()
                targets_u_global_oh = F.one_hot(targets_u_global, args.num_classes).float()

                delta_c = torch.clamp(torch.abs(max_probs_local - max_probs_global) + 1e-6, min=1e-6, max=1.0)
                lambda_dyn = torch.clamp(torch.exp((-math.log(2.0)/0.05) * delta_c), min=1e-6, max=1.0)

                final_targets_u = torch.where(
                    mask_local.unsqueeze(1).bool(),
                    lambda_dyn.unsqueeze(1) * targets_u_local_oh + (1 - lambda_dyn).unsqueeze(1) * targets_u_global_oh,
                    targets_u_global_oh
                )
                
                Lu = (F.kl_div((torch.softmax(logits_u_s, dim=-1)+1e-10).log(), final_targets_u+1e-10, reduction='none').sum(-1) * torch.max(mask_local, mask_global)).mean()
                loss = Lx + args.lambda_u * Lu

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return {k: v.cpu() for k, v in self.local_model.state_dict().items()}

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
        self.base_dataset, self.indices, self.transform = base_dataset, indices, transform
        self.valid_indices = [i for i in indices if i < len(base_dataset)]
        self.targets = [base_dataset.targets[i] for i in self.valid_indices]
    def __len__(self): return len(self.valid_indices)
    def __getitem__(self, idx):
        try:
            img, label = self.base_dataset[self.valid_indices[idx]]
            return self.transform(img) if self.transform else img, label
        except: return self.__getitem__((idx + 1) % len(self.valid_indices))

def main_loop(alpha):
    args = args_parser()
    args.alpha, args.s3_bucket = alpha, 'sage-ham10k-eda'
    exp_name = get_exp_name(args)
    local_ckpt_dir = os.path.join(args.checkpoint_dir, exp_name)

    os.makedirs(f'./results/{args.dataset}/logs', exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=f'./results/{args.dataset}/logs/{exp_name}.log', filemode='a')

    if args.dataset == 'HAM10000':
        args.num_classes, args.num_rounds = 7, 300
        full_dataset = ImageFolder(root=args.path_ham10000)
        train_idx, test_idx = train_test_split(range(len(full_dataset)), test_size=0.20, stratify=full_dataset.targets, random_state=args.seed)

        data_local_training = _SubsetImageFolder(full_dataset, train_idx)
        data_global_test = _SubsetImageFolder(ImageFolder(root=args.path_ham10000, transform=transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.763, 0.545, 0.570], [0.140, 0.152, 0.169])
        ])), test_idx)

    os.makedirs(local_ckpt_dir, exist_ok=True)
    list_label2indices = classify_label(data_local_training, args.num_classes)
    l_lab, l_unlab = partition_train(list_label2indices, args.num_labeled)
    
    if alpha == 0:
        c_lab, c_unlab = clients_indices_homo(l_lab, args.num_classes, args.num_clients), clients_indices_homo(l_unlab, args.num_classes, args.num_clients)
    else:
        c_lab, c_unlab = clients_indices(l_lab, args.num_classes, args.num_clients, alpha, 0), clients_indices(l_unlab, args.num_classes, args.num_clients, alpha, 0)

    for i in range(len(c_lab)):
        if i < len(c_unlab): c_unlab[i].extend(c_lab[i])
        else: c_unlab.append(list(c_lab[i]))
          
    global_model, local_model = Global(args), Local(args)
    start_round, metrics_history = load_checkpoint(global_model.model, local_ckpt_dir, args)

    idx_labeled, idx_unlabeled = Indices2Dataset_labeled(data_local_training), Indices2Dataset_unlabeled_fixmatch(data_local_training)

    for r in tqdm(range(start_round, args.num_rounds + 1), desc='Server'):
        dict_global_params = global_model.download_params()
        online_clients = np.random.RandomState(args.seed + r).choice(range(args.num_clients), args.num_online_clients, replace=False)

        list_dicts_local_params, list_nums_local_data = [], []

        for client in online_clients:
            idx_labeled.load(c_lab[client])
            idx_unlabeled.load(c_unlab[client])
            list_nums_local_data.append(len(c_lab[client]) + len(c_unlab[client]))
          
            list_dicts_local_params.append(local_model.fixmatch_train(args, idx_labeled, idx_unlabeled, copy.deepcopy(dict_global_params), r))
            torch.cuda.empty_cache()

        fedavg_params = global_model.initialize_for_model_fusion(args, list_dicts_local_params, list_nums_local_data, dict_global_params)
        global_model.model.load_state_dict(fedavg_params)

        acc, acsa, macro_f1 = global_model.fedavg_eval(copy.deepcopy(fedavg_params), data_global_test, args.batch_size_test, args)
        metrics_history['acc'].append(acc); metrics_history['acsa'].append(acsa); metrics_history['f1'].append(macro_f1)

        print(f"\n[Round {r:>4}/{args.num_rounds}] Acc: {acc:.4f} | ACSA: {acsa:.4f} | F1: {macro_f1:.4f} ║ Best ACSA: {max(metrics_history['acsa']):.4f}")
        save_checkpoint(r, global_model.download_params(), metrics_history, local_ckpt_dir, args)

        pd.DataFrame({'round': range(1, len(metrics_history['acc']) + 1), 'acc': metrics_history['acc'], 'acsa': metrics_history['acsa']}).to_csv(f'./results/{args.dataset}/STFL_{args.aggregation_method}_alpha={alpha}.csv', index=False)

if __name__ == '__main__':
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    np.random.seed(7)
    args = args_parser()
    main_loop(args.alpha)
