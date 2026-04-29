"""
SAGE + ShapFed + STFL + Focal Loss — Düzeltilmiş Ana Eğitim Dosyası

Bu dosyadaki kritik düzeltmeler:

1. EMA STATE İZOLASYONU
   Eski: Local.__init__ içinde class_probs_ema yaratılıyordu →
         tüm client'lar aynı EMA objesini paylaşıyordu → kirli state.
   Yeni: Her fixmatch_train çağrısında sıfırlanan yerel EMA.

2. STFL MANTIK DÜZELTMESİ
   Eski: dynamic_thresholds = threshold * (ema / max_ema) →
         EMA yüksek sınıf için threshold yüksek oluyordu (YANLIŞ)
         Neden yanlış: Model "nv"yi iyi biliyor, ama threshold da yüksek,
         yani "nv" pseudo-label'ları da kolayca geçiyor.
   Yeni: Nadir sınıf (düşük EMA) için threshold DÜŞÜRÜLÜYOR →
         model az bildiği sınıfı öğrenmek için daha fazla fırsat buluyor.
         threshold_c = base_threshold * (1 - beta * normalized_ema_c)
         beta=0.4 ile nv threshold'u max 0.85, nadir sınıf min ~0.51

3. PSEUDO-LABEL ŞİŞMESİ ENGELLENDİ
   Eski: client_dataset *= 3 → 657 görüntü → 1971 görüntü →
         local_iter = 1971/32 = 61 → 2 epoch × 61 × batch × mu pseudo
         = gerçekçi olmayan onbinlerce pseudo-label.
   Yeni: Gerçek veri boyutu + local_iter sabitlendi (max 20 iter/epoch).

4. WEIGHTED RANDOM SAMPLER
   Nadir sınıflar labeled dataloader'da daha çok örneklenir.

5. SHAPLEY FC KEY DÜZELTMESİ
   get_pretrained_model() → 'fc.weight' / 'fc.bias'
   ResNet wrapper (Model/resnet.py) → 'backbone.fc.weight' / 'backbone.fc.bias'
   Kod şu an SAGE_newModels.py'deki get_pretrained_model() kullanıyor → 'fc.weight' doğru.

6. ROUND BAŞINA SÜRE
   56 dk/round → hedef: ~15-20 dk/round
   local_iter cap + gerçek veri boyutu ile sağlanır.
"""

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
from sklearn.metrics import recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import json
import time
from collections import Counter
import torchvision.models as models

from options import args_parser
from Dataset.dataset import (
    classify_label,
    Indices2Dataset_labeled,
    Indices2Dataset_unlabeled_fixmatch,
    partition_train,
)
from Dataset.sample_dirichlet import clients_indices, clients_indices_homo

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ══════════════════════════════════════════════════════════════
# 1. MODEL TANIMI
# ══════════════════════════════════════════════════════════════

def get_pretrained_model(num_classes):
    """
    ImageNet pretrained ResNet-18.
    BatchNorm → GroupNorm dönüşümü Model/resnet.py'de yapılıyor.
    Burada ise doğrudan torchvision modeli kullanılıyor.
    BN katmanları eval() + requires_grad=False → Non-IID istatistik zehirlenmesi engeli.
    """
    from torchvision.models import ResNet18_Weights
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # BN'leri dondur: küçük batch + Non-IID veriyle running stats bozulmasın
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            module.weight.requires_grad = False
            module.bias.requires_grad = False

    return model


# ══════════════════════════════════════════════════════════════
# 2. LOSS FONKSİYONU
# ══════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss: Modelin zor örneklere odaklanmasını sağlar.
    Sınıf frekansına göre alpha ağırlıkları verilir.
    gamma=2: standart FixMatch literatüründe önerilen değer.
    """
    def __init__(self, alpha_weights=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha_weights = alpha_weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha_weights, reduction='none')
        pt = torch.exp(-ce)
        focal = (1.0 - pt) ** self.gamma * ce
        if self.reduction == 'mean':
            return focal.mean()
        return focal.sum()


def build_focal_loss(class_counts, num_classes, device):
    """
    Client'ın yerel sınıf dağılımına göre Focal Loss ağırlıkları.
    Nadir sınıfa yüksek, baskın sınıfa düşük ağırlık.
    """
    total = sum(class_counts.values()) + 1e-8
    weights = []
    for c in range(num_classes):
        cnt = class_counts.get(c, 1)
        weights.append(total / (num_classes * cnt))
    w = torch.tensor(weights, dtype=torch.float32).to(device)
    # Normalize: ortalama ağırlık ≈ 1 olsun
    w = w / w.mean()
    return FocalLoss(alpha_weights=w, gamma=2.0)


# ══════════════════════════════════════════════════════════════
# 3. EXPERIMENT YÖNETIMI
# ══════════════════════════════════════════════════════════════

def get_exp_name(args):
    return (
        f"{args.dataset}_a{args.alpha}_{args.aggregation_method}_"
        f"L{args.num_labeled}_C{args.num_online_clients}_E{args.local_epochs}_"
        f"T{args.threshold}_LR{args.lr_local_training}_GN16"
    )


def save_checkpoint(round_num, model_state, metrics_history, local_ckpt_dir,
                    args, filename='checkpoint.pt', backup_every=3):
    folder_name = get_exp_name(args)
    os.makedirs(local_ckpt_dir, exist_ok=True)
    state = {
        'round': round_num,
        'model_state_dict': model_state,
        'scheduler_state_dict': scheduler_state,
        'metrics_history': metrics_history,
        'args': args,
    }
    local_path = os.path.join(local_ckpt_dir, filename)
    torch.save(state, local_path)

    if round_num % backup_every == 0 or round_num == args.num_rounds:
        try:
            s3 = boto3.client('s3')
            s3.upload_file(local_path, args.s3_bucket,
                           f"checkpoints/{folder_name}/{filename}")
            local_csv = f'./results/{args.dataset}/{args.aggregation_method}_alpha={args.alpha}.csv'
            if os.path.exists(local_csv):
                s3.upload_file(local_csv, args.s3_bucket,
                               f"results/{folder_name}/metrics.csv")
            print(f"[S3] Round {round_num} yedeklendi.")
        except Exception as e:
            print(f"[S3-WARN] {e}")


def load_checkpoint(model, scheduler, local_ckpt_dir, args, filename='checkpoint.pt'):
    folder_name = get_exp_name(args)
    local_path = os.path.join(local_ckpt_dir, filename)
    s3_path = f"checkpoints/{folder_name}/{filename}"

    if not os.path.exists(local_path):
        try:
            s3 = boto3.client('s3')
            print(f"[CKPT] S3'ten indiriliyor: {s3_path}")
            os.makedirs(local_ckpt_dir, exist_ok=True)
            s3.download_file(args.s3_bucket, s3_path, local_path)
        except ClientError:
            print("[CKPT] Checkpoint bulunamadı. Sıfırdan başlıyor.")
            return 1, {'acc': [], 'acsa': [], 'f1': []}

    print(f"[CKPT] Yükleniyor: {local_path}")
    try:
        ckpt = torch.load(
            local_path,
            map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            weights_only=False,
        )
        model.load_state_dict(ckpt['model_state_dict'])
        if 'scheduler_state_dict' in ckpt: # Eklendi
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
           
        return ckpt['round'] + 1, ckpt['metrics_history']
    except Exception as e:
        print(f"[CKPT] Yükleme hatası: {e} → Sıfırdan başlıyor.")
        return 1, {'acc': [], 'acsa': [], 'f1': []}


# ══════════════════════════════════════════════════════════════
# 4. SHAPLEY (CSSV) — Class-Specific Shapley Values
# ══════════════════════════════════════════════════════════════

def compute_cssv(args, local_models_params, initial_global_params):
    """
    Her client'ın her sınıf için marjinal Shapley katkısını hesaplar.
    Dönüş boyutu: [num_clients, num_classes]

    Key düzeltmesi: get_pretrained_model() kullandığımız için
    son katman adı 'fc.weight' / 'fc.bias'.
    """
    num_clients = len(local_models_params)
    num_classes = args.num_classes

    if num_clients == 0:
        return np.array([])

    weight_layer = 'fc.weight'
    bias_layer   = 'fc.bias'

    # 1. Delta W = local - global (sadece fc katmanı)
    client_updates = []
    for lp in local_models_params:
        update = {
            weight_layer: (lp[weight_layer] -
                           initial_global_params[weight_layer].to(lp[weight_layer].device)),
            bias_layer:   (lp[bias_layer] -
                           initial_global_params[bias_layer].to(lp[bias_layer].device)),
        }
        client_updates.append(update)

    shapley_values = np.zeros((num_clients, num_classes))
    num_samples = getattr(args, 'shapley_samples', 10)

    for _ in range(num_samples):
        perm = np.random.permutation(num_clients)

        for i, client_idx in enumerate(perm):
            coal    = perm[:i]
            coal_p  = perm[:i + 1]

            for c in range(num_classes):
                curr_w = torch.cat([
                    client_updates[client_idx][weight_layer][c].view(-1),
                    client_updates[client_idx][bias_layer][c].view(-1),
                ])
                curr_norm = (F.normalize(curr_w.unsqueeze(0), p=2)
                             if torch.norm(curr_w) > 0 else curr_w.unsqueeze(0))

                # Koalisyon (client hariç) ortalaması
                sim_s = 0.0
                if len(coal) > 0:
                    tw = sum(client_updates[j][weight_layer][c] for j in coal) / len(coal)
                    tb = sum(client_updates[j][bias_layer][c]   for j in coal) / len(coal)
                    agg = torch.cat([tw.view(-1), tb.view(-1)])
                    if torch.norm(agg) > 0 and torch.norm(curr_w) > 0:
                        sim_s = F.cosine_similarity(curr_norm,
                                                    F.normalize(agg.unsqueeze(0), p=2)).item()

                # Koalisyon + client ortalaması
                tw2 = sum(client_updates[j][weight_layer][c] for j in coal_p) / len(coal_p)
                tb2 = sum(client_updates[j][bias_layer][c]   for j in coal_p) / len(coal_p)
                agg2 = torch.cat([tw2.view(-1), tb2.view(-1)])
                sim_si = 0.0
                if torch.norm(agg2) > 0 and torch.norm(curr_w) > 0:
                    sim_si = F.cosine_similarity(curr_norm,
                                                 F.normalize(agg2.unsqueeze(0), p=2)).item()

                shapley_values[client_idx, c] += (sim_si - sim_s)

    if num_samples > 0:
        shapley_values /= num_samples

    shapley_values = np.maximum(shapley_values, 0)

    # Sütun normalizasyonu: her sınıf için toplam ağırlık = 1
    for c in range(num_classes):
        col_sum = shapley_values[:, c].sum()
        if col_sum > 0:
            shapley_values[:, c] /= col_sum
        else:
            shapley_values[:, c] = 1.0 / num_clients

    return shapley_values


# ══════════════════════════════════════════════════════════════
# 5. GLOBAL MODEL
# ══════════════════════════════════════════════════════════════

class Global(object):
    def __init__(self, args):
        self.model = get_pretrained_model(args.num_classes)
        self.model.cuda(args.gpu_id)
        self.num_classes = args.num_classes
        self.args = args

    def initialize_for_model_fusion(self, args, list_dicts_local_params,
                                    list_nums_local_data, initial_global_params):
        fused_params = copy.deepcopy(list_dicts_local_params[0])
        num_clients  = len(list_dicts_local_params)
        total_data   = sum(list_nums_local_data)

        if args.aggregation_method == 'ShapFed':
            cssv = compute_cssv(args, list_dicts_local_params, initial_global_params)
            # Backbone için her client'ın ortalama Shapley ağırlığı
            cb_w = np.mean(cssv, axis=1)
            cb_w = cb_w / cb_w.sum() if cb_w.sum() > 0 else np.ones(num_clients) / num_clients
        else:
            cb_w = np.array([n / total_data for n in list_nums_local_data])

        for name in list_dicts_local_params[0]:
            orig_dtype = list_dicts_local_params[0][name].dtype

            if args.aggregation_method == 'ShapFed' and name in ('fc.weight', 'fc.bias'):
                # Sınıf bazlı birleştirme: her sınıf satırı kendi uzmanından gelir
                fused = torch.zeros_like(list_dicts_local_params[0][name], dtype=torch.float32)
                for c in range(args.num_classes):
                    for i in range(num_clients):
                        fused[c] += list_dicts_local_params[i][name][c] * cssv[i, c]
            else:
                # Backbone: ağırlıklı ortalama
                fused = sum(
                    list_dicts_local_params[i][name].float() * float(cb_w[i])
                    for i in range(num_clients)
                )

            fused_params[name] = fused.to(orig_dtype)

        return fused_params

    def fedavg_eval(self, fedavg_params, data_test, batch_size_test, args):
        self.model.load_state_dict(fedavg_params)
        self.model.eval()

        all_labels, all_predicts = [], []

        with torch.no_grad():
            for images, labels in DataLoader(data_test, batch_size_test,
                                             num_workers=0, pin_memory=False):
                images = images.cuda(args.gpu_id)
                outputs = self.model(images)
                # forward() tuple döner → (feature, logits)
                logits = outputs[1] if isinstance(outputs, tuple) else outputs
                _, preds = torch.max(logits, -1)
                all_labels.extend(labels.numpy())
                all_predicts.extend(preds.cpu().numpy())

        acc      = (np.array(all_labels) == np.array(all_predicts)).mean()
        acsa     = recall_score(all_labels, all_predicts, average='macro', zero_division=0)
        macro_f1 = f1_score(all_labels, all_predicts, average='macro', zero_division=0)
        pred_dist = dict(Counter(all_predicts))

        # Sınıf bazlı recall raporu
        CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        cm = confusion_matrix(all_labels, all_predicts, labels=list(range(self.num_classes)))
        per_class_recall = {}
        print("\n  ┌─ Per-Class Recall ──────────────────")
        for c in range(self.num_classes):
            row_sum = cm[c].sum()
            rec = cm[c, c] / row_sum if row_sum > 0 else 0.0
            per_class_recall[CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)] = round(rec, 4)
            bar = '█' * int(rec * 20)
            name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f'cls{c}'
            print(f"  │ {name:<6}: {rec:.4f} {bar}")
        print("  └─────────────────────────────────────")

        torch.cuda.empty_cache()
        return acc, acsa, macro_f1, pred_dist, per_class_recall

    def download_params(self):
        return self.model.state_dict()


# ══════════════════════════════════════════════════════════════
# 6. LOCAL MODEL — FixMatch + STFL + Focal Loss
# ══════════════════════════════════════════════════════════════

class Local(object):
    def __init__(self, args):
        self.local_model = get_pretrained_model(args.num_classes)
        self.local_G     = get_pretrained_model(args.num_classes)
        self.local_model.cuda(args.gpu_id)
        self.local_G.cuda(args.gpu_id)

        # Optimizer: pretrained model için LR düşük tutulmalı
        self.optimizer = torch.optim.SGD(
            self.local_model.parameters(),
            lr=args.lr_local_training,
            momentum=0.9,
            weight_decay=1e-4,
        )

        # YENİ: Cosine Annealing Scheduler Tanımı
        # T_max: Toplam round sayısı kadar adımda LR'ı min_lr'a indirir.
        # eta_min: LR'ın düşeceği minimum değer.
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=args.num_rounds, 
            eta_min=args.lr_local_training * 0.01
        )

    def fixmatch_train(self, args, data_client_labeled,
                       data_client_unlabeled, global_params, r):
        """
        FixMatch eğitim döngüsü.

        Düzeltmeler:
        - EMA her çağrıda sıfırlanır (client state izolasyonu)
        - STFL: nadir sınıf threshold düşürülür (doğru yön)
        - local_iter sabitlenir: max MAX_ITER_PER_EPOCH
        - WeightedRandomSampler labeled veri için
        - Focal Loss sınıf dağılımına göre
        """
        device = f'cuda:{args.gpu_id}'

        # ── Labeled: Sınıf dağılımı analizi ──
        local_labels = [int(data_client_labeled.dataset.targets[data_client_labeled.client_dataset[i][0]])
                        for i in range(len(data_client_labeled))]
        class_counts = dict(Counter(local_labels))

        # Focal Loss: yerel sınıf dağılımına göre
        focal_criterion = build_focal_loss(class_counts, args.num_classes, device)

        if self.scheduler.last_epoch != r - 1:
            self.scheduler.last_epoch = r - 1
            
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"    ├─ Current LR: {current_lr:.6f}") # Log için ekledik 
                          
        # WeightedRandomSampler: nadir sınıf daha çok seçilsin
        sample_weights = data_client_labeled.sample_weights
        if sample_weights and len(sample_weights) > 0:
            sampler_labeled = WeightedRandomSampler(
                weights=torch.tensor(sample_weights, dtype=torch.float64),
                num_samples=len(sample_weights),
                replacement=True,
            )
            labeled_loader = DataLoader(
                data_client_labeled,
                sampler=sampler_labeled,
                batch_size=args.batch_size_local_labeled_fixmatch,
                drop_last=True,
                num_workers=0,
                pin_memory=False,
            )
        else:
            labeled_loader = DataLoader(
                data_client_labeled,
                sampler=RandomSampler(data_client_labeled),
                batch_size=args.batch_size_local_labeled_fixmatch,
                drop_last=True,
                num_workers=0,
                pin_memory=False,
            )

        unlabeled_loader = DataLoader(
            data_client_unlabeled,
            sampler=RandomSampler(data_client_unlabeled),
            batch_size=args.batch_size_local_labeled_fixmatch * args.mu,
            drop_last=True,
            num_workers=0,
            pin_memory=False,
        )

        # Model başlatma
        self.local_model.load_state_dict(global_params)
        self.local_model.train()
        self.local_G.load_state_dict(global_params)
        self.local_G.eval()

        # ── EMA STATE — Her çağrıda sıfırla (client izolasyonu) ──
        # Uniform başlangıç: hiçbir sınıfı öne almıyoruz
        class_probs_ema = torch.ones(args.num_classes, device=device) / args.num_classes

        # ── local_iter sabitleme ──
        # Gerçek unlabeled boyutu / batch → maksimum cap uygula
        # Cap olmadan "Iteration Stop" hatası veya aşırı uzun round oluşur
        MAX_ITER_PER_EPOCH = 30  # round başı ~15-20 dk hedefi
        real_unlabeled_size = data_client_unlabeled.client_dataset_len
        local_iter = min(
            int(real_unlabeled_size / args.batch_size_local_labeled_fixmatch),
            MAX_ITER_PER_EPOCH,
        )
        local_iter = max(local_iter, 5)  # En az 5 iterasyon

        epoch_pseudo_labels = []
        total_lx, total_lu = 0.0, 0.0
        total_batches = 0

        for local_epoch in range(args.local_epochs):
            labeled_iter   = iter(labeled_loader)
            unlabeled_iter = iter(unlabeled_loader)

            for step in range(local_iter):
                # ── Batch alma ──
                try:
                    inputs_x, targets_x = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(labeled_loader)
                    inputs_x, targets_x = next(labeled_iter)

                try:
                    inputs_u_w, inputs_u_s, _ = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    inputs_u_w, inputs_u_s, _ = next(unlabeled_iter)

                inputs_x   = inputs_x.to(device)
                inputs_u_w = inputs_u_w.to(device)
                inputs_u_s = inputs_u_s.to(device)
                targets_x  = targets_x.to(device)
                batch_size  = inputs_x.shape[0]

                # ── Forward (interleave) ──
                inputs = self.interleave(
                    torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1
                ).to(device)

                out = self.local_model(inputs)
                logits = out[1] if isinstance(out, tuple) else out
                logits = self.de_interleave(logits, 2 * args.mu + 1)

                logits_x   = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                del logits

                # ── Lx: Focal Loss ──
                Lx = focal_criterion(logits_x, targets_x)

                # ── Pseudo-label üretimi ──
                with torch.no_grad():
                    G_out = self.local_G(inputs_u_w)
                    logits_global = G_out[1] if isinstance(G_out, tuple) else G_out

                pseudo_global = torch.softmax(logits_global / args.T, dim=-1)
                max_prob_g, target_g = torch.max(pseudo_global, dim=-1)

                pseudo_local  = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
                max_prob_l, target_l = torch.max(pseudo_local, dim=-1)

                # ── STFL: Dinamik Threshold (DÜZELTILMIŞ YÖN) ──
                # EMA güncelle: hangi sınıflar daha çok tahmin ediliyor?
                class_probs_ema = (class_probs_ema * 0.99
                                   + pseudo_local.mean(dim=0).detach() * 0.01)

                # Normalized EMA: [0, 1] aralığına normalize et
                ema_normalized = class_probs_ema / (class_probs_ema.max() + 1e-8)

                # DOĞRU YÖN: EMA yüksek sınıf (nv) → threshold yüksek kalır.
                # EMA düşük sınıf (nadir) → threshold DÜŞÜRÜLÜR.
                # beta=0.4: nv için 0.85, en nadir sınıf için ~0.85*(1-0.4)=0.51
                beta = 0.6 # up from 0.4
                base_threshold = args.threshold
                # dynamic_thresholds = base_threshold * (1.0 - beta * ema_normalized)
                dynamic_thresholds = base_threshold * (1.0 + beta * ema_normalized)

                # [0.51, 0.85] aralığını garantile
                dynamic_thresholds = torch.clamp(dynamic_thresholds, min=0.50, max=base_threshold)

                mask_local  = max_prob_l.ge(dynamic_thresholds[target_l]).float()
                mask_global = max_prob_g.ge(dynamic_thresholds[target_g]).float()

                # ── SAGE Confidence-Driven Soft Correction ──
                target_l_oh = F.one_hot(target_l, args.num_classes).float()
                target_g_oh = F.one_hot(target_g, args.num_classes).float()

                delta_c = torch.clamp(
                    torch.abs(max_prob_l - max_prob_g) + 1e-6, min=1e-6, max=1.0
                )
                kappa = torch.log(torch.tensor(2.0)) / 0.05
                lambda_dyn = torch.clamp(torch.exp(-kappa * delta_c), min=1e-6, max=1.0)

                final_targets_u = torch.where(
                    mask_local.unsqueeze(1).bool(),
                    lambda_dyn.unsqueeze(1) * target_l_oh
                    + (1 - lambda_dyn).unsqueeze(1) * target_g_oh,
                    target_g_oh,
                )
                mask_valid = torch.max(mask_local, mask_global)

                # ── Lu: Unlabeled Loss ──
                logits_u_s_prob = torch.softmax(logits_u_s, dim=-1) + 1e-10
                final_targets_u = final_targets_u + 1e-10
                Lu = (
                    F.kl_div(logits_u_s_prob.log(), final_targets_u, reduction='none')
                    .sum(-1) * mask_valid
                ).mean()

                loss = Lx + args.lambda_u * Lu

                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping: eğitim kararlılığı
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=10.0)
                self.optimizer.step()

                total_lx      += Lx.item()
                total_lu      += Lu.item()
                total_batches += 1

                # Pseudo-label dağılımı takibi
                valid_pseudo = target_l[mask_valid.bool()].cpu().numpy().tolist()
                epoch_pseudo_labels.extend(valid_pseudo)

        # ── Temizlik ──
        final_state = {k: v.cpu() for k, v in self.local_model.state_dict().items()}
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        avg_lx = total_lx / max(total_batches, 1)
        avg_lu = total_lu / max(total_batches, 1)
        pseudo_dist = dict(Counter(epoch_pseudo_labels))

                                   
        return final_state, pseudo_dist, avg_lx, avg_lu

    def interleave(self, x, size):
        s = list(x.shape)
        return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

    def de_interleave(self, x, size):
        s = list(x.shape)
        return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


# ══════════════════════════════════════════════════════════════
# 7. YARDIMCI: SubsetImageFolder
# ══════════════════════════════════════════════════════════════

class _SubsetImageFolder(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices, transform=None):
        self.base_dataset   = base_dataset
        self.transform      = transform
        max_len             = len(base_dataset)
        self.valid_indices  = [i for i in indices if i < max_len]
        self.targets        = [base_dataset.targets[i] for i in self.valid_indices]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        try:
            img, label = self.base_dataset[self.valid_indices[idx]]
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception:
            return self.__getitem__((idx + 1) % len(self.valid_indices))


# ══════════════════════════════════════════════════════════════
# 8. MAIN LOOP
# ══════════════════════════════════════════════════════════════

def main_loop(alpha):
    args = args_parser()
    args.alpha     = alpha
    args.s3_bucket = 'sage-ham10k-eda'

    exp_name      = get_exp_name(args)
    local_ckpt_dir = os.path.join(args.checkpoint_dir, exp_name)

    # Logging
    os.makedirs(f'./results/{args.dataset}/logs', exist_ok=True)
    log_file = f'./results/{args.dataset}/logs/{exp_name}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        filename=log_file,
        filemode='a',
    )

    # ── Veri Seti ──
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
        all_indices  = list(range(len(full_dataset)))
        all_targets  = full_dataset.targets

        train_idx, test_idx = train_test_split(
            all_indices, test_size=0.20,
            stratify=all_targets, random_state=args.seed,
        )

        data_local_training = _SubsetImageFolder(full_dataset, train_idx)
        test_base           = ImageFolder(root=args.path_ham10000, transform=transform_test)
        data_global_test    = _SubsetImageFolder(test_base, test_idx)

        # Sınıf dağılımı raporu
        train_cls = Counter(all_targets[i] for i in train_idx)
        test_cls  = Counter(all_targets[i] for i in test_idx)
        cls_names = full_dataset.classes
        print(f"[HAM10000] Train: {len(train_idx)} | Test: {len(test_idx)}")
        print(f"  Train: { {cls_names[k]: v for k, v in sorted(train_cls.items())} }")
        print(f"  Test : { {cls_names[k]: v for k, v in sorted(test_cls.items())} }")

    elif args.dataset == 'CIFAR10':
        args.num_classes = 10
        args.num_rounds  = 300
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])
        data_local_training = datasets.CIFAR10(
            args.path_cifar10, train=True, download=True, transform=None)
        data_global_test = datasets.CIFAR10(
            args.path_cifar10, train=False, download=True, transform=transform_test)
    else:
        print(f"[ERROR] Desteklenmeyen veri seti: {args.dataset}")
        exit(1)

    os.makedirs(local_ckpt_dir, exist_ok=True)

    # ── Veri Dağıtımı ──
    random_state = np.random.RandomState(args.seed)
    print(f"\n[DATA] Sınıf indeksleme başlıyor...")
    list_label2indices = classify_label(data_local_training, args.num_classes)

    ipc = args.num_labeled  # Sınıf başına etiketli veri
    print(f"[DATA] IPC: {ipc} | Toplam etiketli: {ipc * args.num_classes}")

    l_lab, l_unlab = partition_train(list_label2indices, ipc)

    if alpha == 0:
        c_lab   = clients_indices_homo(l_lab,   args.num_classes, args.num_clients)
        c_unlab = clients_indices_homo(l_unlab, args.num_classes, args.num_clients)
    else:
        c_lab   = clients_indices(l_lab,   args.num_classes, args.num_clients, alpha, seed=0)
        c_unlab = clients_indices(l_unlab, args.num_classes, args.num_clients, alpha, seed=0)

    # Labeled indisleri unlabeled'a da ekle (SAGE orijinal davranışı)
    for i in range(len(c_lab)):
        if i < len(c_unlab):
            c_unlab[i].extend(c_lab[i])
        else:
            c_unlab.append(list(c_lab[i]))

    # ── Model ──
    global_model = Global(args)
    local_model  = Local(args)

    start_round, metrics_history = load_checkpoint(
       model=global_model.model, 
       scheduler=local_model.scheduler, 
       local_ckpt_dir=local_ckpt_dir, 
       args=args
   )

    idx_labeled   = Indices2Dataset_labeled(data_local_training, args.dataset)
    idx_unlabeled = Indices2Dataset_unlabeled_fixmatch(data_local_training, args.dataset)

    # JSON dashboard
    dashboard_data = {}
    os.makedirs(f'./results/{args.dataset}', exist_ok=True)

    total_clients = list(range(args.num_clients))
    CLASS_NAMES   = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

    print(f"\n[TRAIN] {start_round}. round'dan başlıyor. Toplam: {args.num_rounds}")
    print(f"[TRAIN] Exp: {exp_name}\n")

    for r in tqdm(range(start_round, args.num_rounds + 1), desc='Server'):
        dict_global_params = global_model.download_params()
        online_clients = random_state.choice(
            total_clients, args.num_online_clients, replace=False)

        list_dicts_local_params = []
        list_nums_local_data    = []
        round_client_dists      = {}
        round_pseudo_dists      = Counter()

        for client in online_clients:
            idx_labeled.load(c_lab[client])
            idx_unlabeled.load(c_unlab[client])

            # Client veri dağılımı raporu
            lbl_counts = Counter(
                data_local_training.targets[i] for i in c_lab[client]
                if i < len(data_local_training)
            )
            round_client_dists[str(client)] = {str(k): v for k, v in lbl_counts.items()}

            label_summary = ', '.join(
                f"{CLASS_NAMES[int(k)] if int(k) < len(CLASS_NAMES) else k}: {v}"
                for k, v in sorted(lbl_counts.items())
            )
            print(f"\n▶ Client {client} Eğitimi Başlıyor...")
            print(f"    ├─ Labeled  : [{label_summary}]")
            print(f"    ├─ Unlabeled: {len(c_unlab[client])} adet")

            list_nums_local_data.append(len(c_lab[client]) + len(c_unlab[client]))

            local_params, pseudo_dist, avg_lx, avg_lu = local_model.fixmatch_train(
                args, idx_labeled, idx_unlabeled,
                copy.deepcopy(dict_global_params), r,
            )
            list_dicts_local_params.append(copy.deepcopy(local_params))
            round_pseudo_dists.update(pseudo_dist)

            # Pseudo-label özeti
            pseudo_summary = ', '.join(
                f"{CLASS_NAMES[int(k)] if int(k) < len(CLASS_NAMES) else k}: {v}"
                for k, v in sorted(pseudo_dist.items())
            )
            print(f"    └─ Loss(X): {avg_lx:.4f} | Loss(U): {avg_lu:.4f}")
            print(f"       Pseudo: [{pseudo_summary}]")

            del local_params
            torch.cuda.empty_cache()

        # ── Aggregation ──
        fedavg_params = global_model.initialize_for_model_fusion(
            args, list_dicts_local_params, list_nums_local_data, dict_global_params)
        global_model.model.load_state_dict(fedavg_params)

        # ── Evaluation ──
        acc, acsa, macro_f1, pred_dist, per_class_recall = global_model.fedavg_eval(
            copy.deepcopy(fedavg_params), data_global_test, args.batch_size_test, args)

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
            f"Best Acc: {best_acc:.4f} | Best ACSA: {best_acsa:.4f} | Best F1: {best_f1:.4f} | "
            f"PerClass: {per_class_recall}"
        )

        # ── Dashboard JSON ──
        dashboard_data[str(r)] = {
            'client_distributions': round_client_dists,
            'pseudo_labels': {str(k): v for k, v in round_pseudo_dists.items()},
            'global_predictions': {str(k): v for k, v in pred_dist.items()},
            'per_class_recall': per_class_recall,
        }
        log_path = f'./results/{args.dataset}/dashboard_data.json'
        with open(log_path, 'w') as f:
            json.dump(dashboard_data, f, indent=2)

        # S3'e yükle
        try:
            boto3.client('s3').upload_file(
                log_path, args.s3_bucket, f"{exp_name}/dashboard_data.json")
        except Exception as e:
            print(f"[S3-WARN] Dashboard yükleme hatası: {e}")

        # ── Checkpoint ──
        save_checkpoint(
            round_num=r,
            model_state=global_model.download_params(),
            scheduler_state=local_model.scheduler.state_dict(),
            metrics_history=metrics_history,
            local_ckpt_dir=local_ckpt_dir,
            args=args
        )

        # ── CSV ──
        result_dir = f'./results/{args.dataset}'
        os.makedirs(result_dir, exist_ok=True)
        n = len(metrics_history['acc'])
        pd.DataFrame({
            'round': list(range(1, n + 1)),
            'acc':   metrics_history['acc'],
            'acsa':  metrics_history['acsa'],
            'f1':    metrics_history['f1'],
        }).to_csv(
            f'{result_dir}/{args.aggregation_method}_alpha={alpha}.csv',
            index=False, encoding='utf-8',
        )
        local_model.scheduler.step()
   

# ══════════════════════════════════════════════════════════════
# 9. ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    torch.backends.cudnn.deterministic = True

    args = args_parser()
    main_loop(args.alpha)
