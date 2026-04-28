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
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from sklearn.metrics import recall_score, f1_score
from sklearn.model_selection import train_test_split
import json
import math
from collections import Counter, defaultdict
import torchvision.models as models

# Custom Modülleriniz (Repo'dan gelen)
from Model.resnet import ResNet
from options import args_parser
from Dataset.dataset import (classify_label, show_clients_data_distribution,
                             Indices2Dataset_labeled,
                             Indices2Dataset_unlabeled_fixmatch, partition_train)
from Dataset.sample_dirichlet import clients_indices, clients_indices_homo

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ══════════════════════════════════════════════════════════════
#  MODEL VE LOSS FONKSIYONLARI (GÜNCELLENDİ)
# ══════════════════════════════════════════════════════════════

def get_pretrained_model(num_classes):
    """
    Pretrained ResNet-18 yükler ve BatchNorm katmanlarını GroupNorm(16) ile değiştirir.
    Bu sayede Non-IID verideki istatistik çökmesi engellenir.
    """
    from torchvision.models import ResNet18_Weights
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # BatchNorm'ları tamamen GroupNorm ile değiştiriyoruz (Eski eval() mantığı silindi)
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            num_channels = module.num_features
            gn = nn.GroupNorm(16, num_channels) # GN16
            
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[-1]
            parent = dict(model.named_modules())[parent_name] if parent_name else model
            setattr(parent, child_name, gn)
            
    return model

def focal_loss(inputs, targets, alpha_weights=None, gamma=2):
    """Sınıf ağırlıklı Focal Loss"""
    BCE_loss = F.cross_entropy(inputs, targets, reduction='none', weight=alpha_weights)
    pt = torch.exp(-BCE_loss)
    F_loss = (1-pt)**gamma * BCE_loss
    return F_loss.mean()
  
def get_exp_name(args):
    return (f"{args.dataset}_a{args.alpha}_{args.aggregation_method}_"
            f"L{args.num_labeled}_C{args.num_online_clients}_E{args.local_epochs}_"
            f"T{args.threshold}_LR{args.lr_local_training}_GN16")
  
def save_checkpoint(round_num, model_state, metrics_history, local_ckpt_dir, 
                    args, filename='checkpoint.pt', backup_every=3):
    
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
    
    if round_num % backup_every == 0 or round_num == args.num_rounds:
        s3 = boto3.client('s3')
        s3_ckpt_path = f"checkpoints/{folder_name}/{filename}"
        s3_csv_path  = f"results/{folder_name}/{args.aggregation_method}_alpha={args.alpha}.csv"
        
        try:
            s3.upload_file(local_path, args.s3_bucket, s3_ckpt_path)
            local_csv_path = f'./results/{args.dataset}/{args.aggregation_method}_alpha={args.alpha}.csv'
            if os.path.exists(local_csv_path):
                s3.upload_file(local_csv_path, args.s3_bucket, s3_csv_path)
            print(f"[S3-SYNC] Round {round_num} verileri S3'e yedeklendi.")
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
            print("No checkpoint found on S3. Starting fresh.")
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
  
    if num_clients == 0:
        return np.array([])

    weight_layer = 'fc.weight'
    bias_layer   = 'fc.bias'
  
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
            coalition_indices      = permutation[:i]
            coalition_plus_indices = permutation[:i+1]

            for c in range(num_classes):
                curr_w_c = torch.cat([
                    client_updates[client_idx][weight_layer][c].view(-1),
                    client_updates[client_idx][bias_layer][c].view(-1)
                ])
                
                if torch.norm(curr_w_c) == 0:
                    curr_w_norm = curr_w_c.unsqueeze(0)
                else:
                    curr_w_norm = F.normalize(curr_w_c.unsqueeze(0), p=2)

                sim_s = 0.0
                if len(coalition_indices) > 0:
                    tw_c = torch.zeros_like(client_updates[0][weight_layer][c])
                    tb_c = torch.zeros_like(client_updates[0][bias_layer][c])
                    for co_idx in coalition_indices:
                        tw_c += client_updates[co_idx][weight_layer][c]
                        tb_c += client_updates[co_idx][bias_layer][c]
                    tw_c /= len(coalition_indices)
                    tb_c /= len(coalition_indices)
                    
                    agg_c = torch.cat([tw_c.view(-1), tb_c.view(-1)])
                    if torch.norm(agg_c) > 0 and torch.norm(curr_w_c) > 0:
                        sim_s = F.cosine_similarity(curr_w_norm, F.normalize(agg_c.unsqueeze(0), p=2)).item()

                tw2_c = torch.zeros_like(client_updates[0][weight_layer][c])
                tb2_c = torch.zeros_like(client_updates[0][bias_layer][c])
                for co_idx in coalition_plus_indices:
                    tw2_c += client_updates[co_idx][weight_layer][c]
                    tb2_c += client_updates[co_idx][bias_layer][c]
                tw2_c /= len(coalition_plus_indices)
                tb2_c /= len(coalition_plus_indices)
                
                agg2_c = torch.cat([tw2_c.view(-1), tb2_c.view(-1)])
                sim_si = 0.0
                if torch.norm(agg2_c) > 0 and torch.norm(curr_w_c) > 0:
                    sim_si = F.cosine_similarity(curr_w_norm, F.normalize(agg2_c.unsqueeze(0), p=2)).item()

                shapley_values[client_idx, c] += (sim_si - sim_s)

    if num_samples > 0:
        shapley_values /= num_samples

    shapley_values = np.maximum(shapley_values, 0)
    
    for c in range(num_classes):
        col_sum = np.sum(shapley_values[:, c])
        if col_sum > 0:
            shapley_values[:, c] /= col_sum
        else:
            shapley_values[:, c] = 1.0 / num_clients

    return shapley_values


# ══════════════════════════════════════════════════════════════
#  GLOBAL MODEL
# ══════════════════════════════════════════════════════════════

class Global(object):
    def __init__(self, args):
        self.model = get_pretrained_model(args.num_classes)
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
                list_values_param = []
                for i in range(num_clients):
                    list_values_param.append(list_dicts_local_params[i][name_param] * list_nums_local_data[i])
                fused_tensor = sum(list_values_param) / total_data

            else:
                if name_param == 'fc.weight' or name_param == 'fc.bias':
                    fused_tensor = torch.zeros_like(list_dicts_local_params[0][name_param], dtype=torch.float32)
                    for c in range(args.num_classes):
                        for i in range(num_clients):
                            w_c = cssv_weights[i, c]
                            fused_tensor[c] += list_dicts_local_params[i][name_param][c] * w_c
                else:
                    fused_tensor = sum(list_dicts_local_params[i][name_param] * client_backbone_weights[i] 
                                       for i in range(num_clients))

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
            for images, labels in DataLoader(data_test, batch_size_test, num_workers=0):
                images, labels = images.cuda(args.gpu_id), labels.cuda(args.gpu_id)
                outputs = self.model(images)
                _, predicts = torch.max(outputs, -1)
                num_corrects += torch.sum(torch.eq(predicts.cpu(), labels.cpu())).item()
                all_labels.extend(labels.cpu().numpy())
                all_predicts.extend(predicts.cpu().numpy())
        
        accuracy = num_corrects / len(data_test)
        acsa = recall_score(all_labels, all_predicts, average='macro', zero_division=0)
        macro_f1 = f1_score(all_labels, all_predicts, average='macro', zero_division=0)

        pred_dist = dict(Counter(all_predicts)) 
        
        torch.cuda.empty_cache()
        return accuracy, acsa, macro_f1, pred_dist

    def download_params(self):
        return self.model.state_dict()
      
# ══════════════════════════════════════════════════════════════
#  LOCAL MODEL (EMA IZOLASYONU VE OVERSAMPLING EKLENDI)
# ══════════════════════════════════════════════════════════════

# class Local(object):
#     def __init__(self, args, client_id):
#         self.client_id = client_id
#         self.local_model = get_pretrained_model(args.num_classes)
#         self.local_G = get_pretrained_model(args.num_classes)
      
#         self.local_model.cuda(args.gpu_id)
#         self.local_G.cuda(args.gpu_id)
#         self.optimizer = torch.optim.SGD(
#             self.local_model.parameters(),
#             lr=args.lr_local_training, momentum=0.9, weight_decay=1e-4
#         )

#     def fixmatch_train(self, args, data_client_labeled, data_client_unlabeled, global_params, r, current_ema):
        
#         # 1. Oversampling (Sınıf dengesizliğini gidermek için)
#         local_labels = [int(data_client_labeled[i][1]) for i in range(len(data_client_labeled))]
#         class_counts = np.bincount(local_labels, minlength=args.num_classes)
#         class_counts_safe = np.where(class_counts == 0, 1, class_counts) # 0'a bölmeyi engelle
        
#         # Focal Loss için ağırlıklar
#         weights = torch.tensor([sum(class_counts)/(args.num_classes*c) for c in class_counts_safe]).float().cuda(args.gpu_id)
        
#         class_weights = 1.0 / class_counts_safe
#         sample_weights = class_weights[local_labels]
#         labeled_sampler = WeightedRandomSampler(
#             weights=torch.tensor(sample_weights, dtype=torch.float64), 
#             num_samples=len(sample_weights), 
#             replacement=True
#         )

#         self.labeled_trainloader = DataLoader(
#             dataset=data_client_labeled,
#             sampler=labeled_sampler,
#             batch_size=args.batch_size_local_labeled_fixmatch,
#             drop_last=True, num_workers=0, pin_memory=True # Deadlock'ı önlemek için num_workers=0
#         )
#         self.unlabeled_trainloader = DataLoader(
#             dataset=data_client_unlabeled,
#             sampler=RandomSampler(data_client_unlabeled),
#             batch_size=args.batch_size_local_labeled_fixmatch * args.mu,
#             drop_last=True, num_workers=0, pin_memory=True
#         )
#         self.local_model.load_state_dict(global_params)
#         self.local_model.train()
#         self.local_G.load_state_dict(global_params)
#         self.local_G.eval()

#         epoch_pseudo_labels = []
        
#         # EMA State'i client'ın kendi hafızasından al
#         class_probs_ema = current_ema.cuda(args.gpu_id)

#         for local_epoch in range(args.local_epochs):
#             labeled_iter   = iter(self.labeled_trainloader)
#             unlabeled_iter = iter(self.unlabeled_trainloader)
#             local_iter = int(len(data_client_unlabeled) / args.batch_size_local_labeled_fixmatch)

#             for epoch in range(local_iter):
#                 try: inputs_x, targets_x = next(labeled_iter)
#                 except StopIteration: 
#                     labeled_iter = iter(self.labeled_trainloader)
#                     inputs_x, targets_x = next(labeled_iter)

#                 try: inputs_u_w, inputs_u_s, targets_u_gt = next(unlabeled_iter)
#                 except StopIteration: 
#                     unlabeled_iter = iter(self.unlabeled_trainloader)
#                     inputs_u_w, inputs_u_s, targets_u_gt = next(unlabeled_iter)

#                 inputs_x   = inputs_x.cuda(args.gpu_id)
#                 inputs_u_w = inputs_u_w.cuda(args.gpu_id)
#                 inputs_u_s = inputs_u_s.cuda(args.gpu_id)
#                 targets_x  = targets_x.cuda(args.gpu_id)
#                 batch_size = inputs_x.shape[0]

#                 inputs = self.interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1).cuda(args.gpu_id)

#                 logits = self.local_model(inputs)
#                 logits    = self.de_interleave(logits, 2 * args.mu + 1)
#                 logits_x  = logits[:batch_size]
#                 logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                
#                 # Ağırlıklı Focal Loss kullanıyoruz
#                 Lx = focal_loss(logits_x, targets_x, alpha_weights=weights) 

#                 with torch.no_grad():
#                     logits_u_w_global = self.local_G(inputs_u_w)
                    
#                 pseudo_label_global  = torch.softmax(logits_u_w_global.detach() / args.T, dim=-1)
#                 max_probs_global, targets_u_global = torch.max(pseudo_label_global, dim=-1)

#                 pseudo_label_local = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
#                 max_probs_local, targets_u_local = torch.max(pseudo_label_local, dim=-1)

#                 # STFL: Client izole edilmiş EMA güncellemesi
#                 class_probs_ema = class_probs_ema * 0.999 + pseudo_label_local.mean(dim=0) * 0.001
#                 max_ema = class_probs_ema.max()
#                 dynamic_thresholds = args.threshold * (class_probs_ema / max_ema)

#                 targets_u_local_one_hot  = F.one_hot(targets_u_local,  args.num_classes).float()
#                 targets_u_global_one_hot = F.one_hot(targets_u_global, args.num_classes).float()

#                 # Sabit threshold yerine dynamic_thresholds kullanıldı
#                 mask_local  = max_probs_local.ge(dynamic_thresholds[targets_u_local]).float()
#                 mask_global = max_probs_global.ge(dynamic_thresholds[targets_u_global]).float()

#                 delta_c = torch.clamp(torch.abs(max_probs_local - max_probs_global) + 1e-6, min=1e-6, max=1.0)
#                 kappa          = torch.log(torch.tensor(2.0)) / 0.05
#                 lambda_dynamic = torch.clamp(torch.exp(-kappa * delta_c), min=1e-6, max=1.0)

#                 final_targets_u = torch.where(
#                     mask_local.unsqueeze(1).bool(),
#                     lambda_dynamic.unsqueeze(1) * targets_u_local_one_hot + (1 - lambda_dynamic).unsqueeze(1) * targets_u_global_one_hot,
#                     targets_u_global_one_hot
#                 )
#                 mask_valid = torch.max(mask_local, mask_global)

#                 valid_pseudo = targets_u_local[mask_valid.bool()].cpu().numpy().tolist()
#                 epoch_pseudo_labels.extend(valid_pseudo)

#                 logits_u_s_probs = torch.softmax(logits_u_s, dim=-1) + 1e-10
#                 final_targets_u  = final_targets_u + 1e-10

#                 Lu   = (F.kl_div(logits_u_s_probs.log(), final_targets_u, reduction='none').sum(-1) * mask_valid).mean()
#                 loss = Lx + args.lambda_u * Lu

#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()

#         final_state = {k: v.cpu() for k, v in self.local_model.state_dict().items()}
#         self.optimizer.zero_grad(set_to_none=True)
#         pseudo_dist = dict(Counter(epoch_pseudo_labels))
        
#         # Güncellenmiş EMA'yı CPU'da döndürüyoruz ki bir sonraki round kullanılabilsin
#         return final_state, pseudo_dist, class_probs_ema.cpu()

# ══════════════════════════════════════════════════════════════
#  LOCAL MODEL (EMA IZOLASYONU, OVERSAMPLING VE CONFIDENCE TAKİBİ)
# ══════════════════════════════════════════════════════════════

class Local(object):
    def __init__(self, args, client_id):
        self.client_id = client_id
        self.local_model = get_pretrained_model(args.num_classes)
        self.local_G = get_pretrained_model(args.num_classes)
      
        self.local_model.cuda(args.gpu_id)
        self.local_G.cuda(args.gpu_id)
        self.optimizer = torch.optim.SGD(
            self.local_model.parameters(),
            lr=args.lr_local_training, momentum=0.9, weight_decay=1e-4
        )

    def fixmatch_train(self, args, data_client_labeled, data_client_unlabeled, global_params, r, current_ema):
        
        # 1. Oversampling (Sınıf dengesizliğini gidermek için)
        local_labels = [int(data_client_labeled[i][1]) for i in range(len(data_client_labeled))]
        class_counts = np.bincount(local_labels, minlength=args.num_classes)
        class_counts_safe = np.where(class_counts == 0, 1, class_counts) 
        
        # Focal Loss için ağırlıklar
        weights = torch.tensor([sum(class_counts)/(args.num_classes*c) for c in class_counts_safe]).float().cuda(args.gpu_id)
        
        class_weights = 1.0 / class_counts_safe
        sample_weights = class_weights[local_labels]
        labeled_sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float64), 
            num_samples=len(sample_weights), 
            replacement=True
        )

        self.labeled_trainloader = DataLoader(
            dataset=data_client_labeled, sampler=labeled_sampler,
            batch_size=args.batch_size_local_labeled_fixmatch,
            drop_last=True, num_workers=0, pin_memory=True 
        )
        self.unlabeled_trainloader = DataLoader(
            dataset=data_client_unlabeled, sampler=RandomSampler(data_client_unlabeled),
            batch_size=args.batch_size_local_labeled_fixmatch * args.mu,
            drop_last=True, num_workers=0, pin_memory=True
        )
        self.local_model.load_state_dict(global_params)
        self.local_model.train()
        self.local_G.load_state_dict(global_params)
        self.local_G.eval()

        epoch_pseudo_labels = []
        epoch_pseudo_confidences = defaultdict(list) # YENİ: Güven skorlarını tutacağımız sözlük
        class_probs_ema = current_ema.cuda(args.gpu_id)

        # TAKİP DEĞİŞKENLERİ 
        total_lx, total_lu, correct_preds, total_samples = 0.0, 0.0, 0, 0

        for local_epoch in range(args.local_epochs):
            labeled_iter   = iter(self.labeled_trainloader)
            unlabeled_iter = iter(self.unlabeled_trainloader)
            local_iter = int(len(data_client_unlabeled) / args.batch_size_local_labeled_fixmatch)

            for epoch in range(local_iter):
                try: inputs_x, targets_x = next(labeled_iter)
                except StopIteration: 
                    labeled_iter = iter(self.labeled_trainloader)
                    inputs_x, targets_x = next(labeled_iter)

                try: inputs_u_w, inputs_u_s, targets_u_gt = next(unlabeled_iter)
                except StopIteration: 
                    unlabeled_iter = iter(self.unlabeled_trainloader)
                    inputs_u_w, inputs_u_s, targets_u_gt = next(unlabeled_iter)

                inputs_x   = inputs_x.cuda(args.gpu_id)
                inputs_u_w = inputs_u_w.cuda(args.gpu_id)
                inputs_u_s = inputs_u_s.cuda(args.gpu_id)
                targets_x  = targets_x.cuda(args.gpu_id)
                batch_size = inputs_x.shape[0]

                inputs = self.interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1).cuda(args.gpu_id)

                features, logits = self.local_model(inputs)
                logits    = self.de_interleave(logits, 2 * args.mu + 1)
                logits_x  = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                
                # Accuracy Hesabı
                preds_x = logits_x.argmax(dim=-1)
                correct_preds += (preds_x == targets_x).sum().item()
                total_samples += targets_x.size(0)

                Lx = focal_loss(logits_x, targets_x, alpha_weights=weights) 

                with torch.no_grad():
                    _, logits_u_w_global = self.local_G(inputs_u_w)
                    
                pseudo_label_global  = torch.softmax(logits_u_w_global.detach() / args.T, dim=-1)
                max_probs_global, targets_u_global = torch.max(pseudo_label_global, dim=-1)

                pseudo_label_local = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
                max_probs_local, targets_u_local = torch.max(pseudo_label_local, dim=-1)

                class_probs_ema = class_probs_ema * 0.999 + pseudo_label_local.mean(dim=0) * 0.001
                max_ema = class_probs_ema.max()
                dynamic_thresholds = args.threshold * (class_probs_ema / max_ema)

                targets_u_local_one_hot  = F.one_hot(targets_u_local,  args.num_classes).float()
                targets_u_global_one_hot = F.one_hot(targets_u_global, args.num_classes).float()

                mask_local  = max_probs_local.ge(dynamic_thresholds[targets_u_local]).float()
                mask_global = max_probs_global.ge(dynamic_thresholds[targets_u_global]).float()

                delta_c = torch.clamp(torch.abs(max_probs_local - max_probs_global) + 1e-6, min=1e-6, max=1.0)
                kappa          = torch.log(torch.tensor(2.0)) / 0.05
                lambda_dynamic = torch.clamp(torch.exp(-kappa * delta_c), min=1e-6, max=1.0)

                final_targets_u = torch.where(
                    mask_local.unsqueeze(1).bool(),
                    lambda_dynamic.unsqueeze(1) * targets_u_local_one_hot + (1 - lambda_dynamic).unsqueeze(1) * targets_u_global_one_hot,
                    targets_u_global_one_hot
                )
                
                # YENİ: Geçerli pseudo labelların güven skorlarını (prob) kaydetme
                mask_valid_bool = torch.max(mask_local, mask_global).bool()
                valid_pseudo = targets_u_local[mask_valid_bool].cpu().numpy().tolist()
                valid_probs = max_probs_local[mask_valid_bool].cpu().numpy().tolist()

                epoch_pseudo_labels.extend(valid_pseudo)
                for cls, prob in zip(valid_pseudo, valid_probs):
                    epoch_pseudo_confidences[cls].append(prob)

                logits_u_s_probs = torch.softmax(logits_u_s, dim=-1) + 1e-10
                final_targets_u  = final_targets_u + 1e-10

                Lu   = (F.kl_div(logits_u_s_probs.log(), final_targets_u, reduction='none').sum(-1) * mask_valid_bool.float()).mean()
                loss = Lx + args.lambda_u * Lu

                total_lx += Lx.item()
                total_lu += Lu.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        final_state = {k: v.cpu() for k, v in self.local_model.state_dict().items()}
        self.optimizer.zero_grad(set_to_none=True)
        pseudo_dist = dict(Counter(epoch_pseudo_labels))
        
        # BİLGİLERİ VE GÜVEN SKORLARINI EKRANA YAZDIR
        local_acc = (correct_preds / total_samples) * 100 if total_samples > 0 else 0
        avg_lx = total_lx / (args.local_epochs * local_iter) if local_iter > 0 else 0
        avg_lu = total_lu / (args.local_epochs * local_iter) if local_iter > 0 else 0
        
        # Sınıf bazlı ortalama güven skorlarını hesapla
        pseudo_conf_avg = {k: (sum(v) / len(v)) * 100 for k, v in epoch_pseudo_confidences.items()}
        
        cls_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
        pseudo_str_parts = []
        for k in sorted(pseudo_dist.keys()):
            count = pseudo_dist[k]
            conf = pseudo_conf_avg[k]
            pseudo_str_parts.append(f"{cls_names[k]}: {count} (%{conf:.1f})")
            
        pseudo_str = ", ".join(pseudo_str_parts)

        print(f"    └─ Bitti! Local Acc: %{local_acc:.1f} | Loss(X): {avg_lx:.3f} | Loss(U): {avg_lu:.3f}")
        if len(pseudo_str) > 0:
            print(f"    └─ Üretilen Pseudo-Labels: [{pseudo_str}]")
        else:
            print(f"    └─ Üretilen Pseudo-Labels: [Hiç üretilmedi]")

        return final_state, pseudo_dist, class_probs_ema.cpu()

    def interleave(self, x, size):
        s = list(x.shape)
        return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

    def de_interleave(self, x, size):
        s = list(x.shape)
        return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


# ══════════════════════════════════════════════════════════════
#  S3 SYNC FONKSIYONU
# ══════════════════════════════════════════════════════════════

def sync_data_from_s3(args):
    print("Skipping S3 sync, using local data on NVMe.")
    pass

# ══════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════

def main_loop(alpha):
    args = args_parser()
    args.alpha = alpha
    args.s3_bucket = 'sage-ham10k-eda'
    sync_data_from_s3(args)

    exp_name = get_exp_name(args)
    local_ckpt_dir = os.path.join(args.checkpoint_dir, exp_name)

    log_dir  = f'./results/{args.dataset}/logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir,f'{exp_name}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='a'
    )

    if args.dataset == 'HAM10000':
        args.num_classes = 7
        args.num_rounds  = 300
        ham_mean = [0.763, 0.545, 0.570]
        ham_std  = [0.140, 0.152, 0.169]

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=ham_mean, std=ham_std),
        ])

        full_dataset = ImageFolder(root=args.path_ham10000, transform=None)

        all_indices = list(range(len(full_dataset)))
        all_targets = [full_dataset.targets[i] for i in all_indices]

        train_indices, test_indices = train_test_split(
            all_indices, test_size=0.20, stratify=all_targets, random_state=args.seed
        )

        data_local_training = _SubsetImageFolder(full_dataset, train_indices)
        test_full           = ImageFolder(root=args.path_ham10000, transform=transform_test)
        data_global_test    = _SubsetImageFolder(test_full, test_indices)

        from collections import Counter
        train_cls = Counter(all_targets[i] for i in train_indices)
        test_cls  = Counter(all_targets[i] for i in test_indices)
        cls_names = full_dataset.classes
        print(f"[HAM10000] Train: {len(train_indices)} | Test: {len(test_indices)}")
        print(f"[HAM10000] Classes: {cls_names}")
        print(f"[HAM10000] Train dist: { {cls_names[k]: v for k,v in sorted(train_cls.items())} }")
        print(f"[HAM10000] Test  dist: { {cls_names[k]: v for k,v in sorted(test_cls.items())} }")
        
    else:
        print(f"[ERROR] AWS senaryosu sadece HAM10000 icin kuruldu. {args.dataset} secildi.")
        exit(1)

    local_ckpt_dir = os.path.join(args.checkpoint_dir,exp_name)
    os.makedirs(local_ckpt_dir, exist_ok=True)
    print(f"Local checkpoint dir : {local_ckpt_dir}")

    random_state = np.random.RandomState(args.seed)
    print("--> Sınıf bazlı indeksleme başlıyor...")
    list_label2indices = classify_label(data_local_training, args.num_classes)
    print(f"--> Sınıflandırma bitti. Labeled/Unlabeled ayrılıyor (Labeled: {args.num_labeled})...")

    ipc = args.num_labeled
    total_labeled = ipc * args.num_classes
    print(f"[INFO] Sınıf Başına Etiket (IPC): {ipc} | Toplam Etiketli Veri: {total_labeled}")
    
    list_label2indices_labeled, list_label2indices_unlabeled = partition_train(list_label2indices, ipc)
  
    print(f"--> Dirichlet Dağılımı hesaplanıyor (Alpha: {alpha})...")

    if alpha == 0:
        list_client2indices_labeled   = clients_indices_homo(list_label2indices_labeled, args.num_classes, args.num_clients)
        list_client2indices_unlabeled = clients_indices_homo(list_label2indices_unlabeled, args.num_classes, args.num_clients)
    else:
        list_client2indices_labeled   = clients_indices(list_label2indices_labeled, args.num_classes, args.num_clients, alpha, seed=0)
        list_client2indices_unlabeled = clients_indices(list_label2indices_unlabeled, args.num_classes, args.num_clients, alpha, seed=0)
        print("--> Dağılım hesaplandı, eğitim başlıyor!")

    print(f"[DEBUG] labeled uzunluk: {len(list_client2indices_labeled)}, unlabeled uzunluk: {len(list_client2indices_unlabeled)}")
    
    for client in range(len(list_client2indices_labeled)):
        if client < len(list_client2indices_unlabeled):
            list_client2indices_unlabeled[client].extend(list_client2indices_labeled[client])
        else:
            list_client2indices_unlabeled.append(list(list_client2indices_labeled[client]))
          
    global_model = Global(args)
    
    start_round, metrics_history = load_checkpoint(global_model.model, local_ckpt_dir, args)

    total_clients          = list(range(args.num_clients))
    indices2data_labeled   = Indices2Dataset_labeled(data_local_training)
    indices2data_unlabeled = Indices2Dataset_unlabeled_fixmatch(data_local_training)

    # İZOLASYON: EMA state'lerini her client için bağımsız tutan dictionary
    client_ema_dict = {i: torch.ones(args.num_classes)/args.num_classes for i in range(args.num_clients)}

    dashboard_data = {} 
    os.makedirs('./results/HAM10000', exist_ok=True) 
  
    for r in tqdm(range(start_round, args.num_rounds + 1), desc='Server'):
        dict_global_params = global_model.download_params()
        online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)

        list_dicts_local_params = []
        list_nums_local_data    = []
        round_client_dists = {} 
        round_pseudo_dists = Counter()

        for client in online_clients:
          
            # Client verilerini say ve isimlendir
            lbl_counts = Counter([data_local_training.targets[i] for i in list_client2indices_labeled[client]])
            round_client_dists[str(client)] = {str(k): v for k, v in lbl_counts.items()}
            
            cls_names = full_dataset.classes if args.dataset == 'HAM10000' else [str(i) for i in range(args.num_classes)]
            dist_str = ", ".join([f"{cls_names[k]}: {v}" for k, v in sorted(lbl_counts.items())])
            unlab_len = len(list_client2indices_unlabeled[client])

            print(f"\n▶ Client {client} Eğitimi Başlıyor...")
            print(f"    ├─ Labeled Veri : [{dist_str}]")
            print(f"    ├─ Unlabeled Veri: {unlab_len} adet")
          
            indices2data_labeled.load(list_client2indices_labeled[client])
            indices2data_unlabeled.load(list_client2indices_unlabeled[client])

            # lbl_counts = Counter([data_local_training.targets[i] for i in list_client2indices_labeled[client]])
            # round_client_dists[str(client)] = {str(k): v for k, v in lbl_counts.items()}

            list_nums_local_data.append(len(list_client2indices_labeled[client]) + len(list_client2indices_unlabeled[client]))
          
            # Local objesi client_id ile her defasında yeni oluşturuluyor
            local_model = Local(args, client)
            
            # Eğitimi başlat ve güncellenmiş EMA'yı geri al
            local_params, pseudo_dist, updated_ema = local_model.fixmatch_train(
                args, indices2data_labeled, indices2data_unlabeled,
                copy.deepcopy(dict_global_params), r, client_ema_dict[client]
            )
            
            # EMA'yı bir sonraki round için sözlüğe kaydet
            client_ema_dict[client] = updated_ema
          
            list_dicts_local_params.append(copy.deepcopy(local_params))
            round_pseudo_dists.update(pseudo_dist) 

            del local_params, local_model
            torch.cuda.empty_cache()

        fedavg_params = global_model.initialize_for_model_fusion(
            args, list_dicts_local_params, list_nums_local_data, dict_global_params
        )
        global_model.model.load_state_dict(fedavg_params)

        acc, acsa, macro_f1, global_pred_dist = global_model.fedavg_eval(
            copy.deepcopy(fedavg_params), data_global_test, args.batch_size_test, args
        )

        dashboard_data[str(r)] = {
            "client_distributions": round_client_dists,
            "pseudo_labels": {str(k): v for k, v in round_pseudo_dists.items()},
            "global_predictions":  {str(k): v for k, v in global_pred_dist.items()} 
        }
        log_path = './results/HAM10000/dashboard_data.json'
        
        with open(log_path, 'w') as f:
            json.dump(dashboard_data, f, indent=4)

        try:
            s3_client = boto3.client('s3')
            s3_bucket = "sage-ham10k-eda"
            s3_key = f"{exp_name}/dashboard_data.json"
            
            s3_client.upload_file(log_path, s3_bucket, s3_key)
            print(f"--> Dashboard verisi S3'e yüklendi: {s3_key}")
        except Exception as e:
            print(f"--> S3 yükleme hatası: {e}")

        metrics_history['acc'].append(acc)
        metrics_history['acsa'].append(acsa)
        metrics_history['f1'].append(macro_f1)

        best_acc  = max(metrics_history['acc'])
        best_acsa = max(metrics_history['acsa'])
        best_f1   = max(metrics_history['f1'])

        print(
            f"\n[Round {r:>4}/{args.num_rounds}]  "
            f"Acc: {acc:.4f} | ACSA: {acsa:.4f} | F1: {macro_f1:.4f}  ║  "
            f"Best → Acc: {best_acc:.4f} | ACSA: {best_acsa:.4f} | F1: {best_f1:.4f}"
        )
        logging.info(
            f"Round {r} | Acc: {acc:.4f} | ACSA: {acsa:.4f} | F1: {macro_f1:.4f} | "
            f"Best Acc: {best_acc:.4f} | Best ACSA: {best_acsa:.4f} | Best F1: {best_f1:.4f}"
        )

        save_checkpoint(
            r, global_model.download_params(), metrics_history,
            local_ckpt_dir=local_ckpt_dir,
            args=args,
            backup_every=3
        )

        result_dir  = f'./results/{args.dataset}'
        os.makedirs(result_dir, exist_ok=True)
        result_file = f'{result_dir}/{args.aggregation_method}_alpha={alpha}.csv'
        n = len(metrics_history['acc'])
        pd.DataFrame({
            'round': list(range(1, n + 1)),
            'acc':   metrics_history['acc'],
            'acsa':  metrics_history['acsa'],
            'f1':    metrics_history['f1'],
        }).to_csv(result_file, index=False, encoding='utf-8')

# ══════════════════════════════════════════════════════════════
#  YARDIMCI
# ══════════════════════════════════════════════════════════════
class _SubsetImageFolder(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices, transform=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
        max_len = len(base_dataset)
        self.valid_indices = [i for i in indices if i < max_len]
        self.targets = [base_dataset.targets[i] for i in self.valid_indices]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        try:
            real_idx = self.valid_indices[idx]
            img, label = self.base_dataset[real_idx]
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self.valid_indices))

# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    torch.backends.cudnn.deterministic = True
    args = args_parser()
    main_loop(args.alpha)
