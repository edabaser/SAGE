# """
# SAGE_final.py — Unified Federated Semi-Supervised Learning
# =============================================================

# Tüm geliştirmeler tek dosyada, flag ile açılıp kapatılır.

# Checkpoint isimlendirme:
#   {dataset}_a{alpha}_{agg}_{model}_{flags}_L{ipc}_C{online}_E{epochs}_T{thr}_LR{lr}
#   Flags: FL STFL WS MA CLR GN EMA SEMA PINIT

# Örnek run komutları → run_commands.sh dosyasına bakın.
# """

# import os, copy, random, logging, json
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from tqdm import tqdm
# import boto3
# from botocore.exceptions import ClientError
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
# from torchvision.datasets import ImageFolder
# from sklearn.metrics import recall_score, f1_score, confusion_matrix
# from sklearn.model_selection import train_test_split
# from collections import Counter
# import torchvision.models as models

# from options import args_parser
# from Dataset.dataset import (
#     classify_label, Indices2Dataset_labeled,
#     Indices2Dataset_unlabeled_fixmatch, partition_train,
# )

# def partition_train_perclass(list_label2indices, ipc_per_class):
#     """
#     Her sinif icin farkli IPC degeri kullanan partition_train.
#     ipc_per_class: [ipc_0, ipc_1, ..., ipc_C] listesi
#     """
#     list_label2indices_labeled   = []
#     list_label2indices_unlabeled = []
#     for c, indices in enumerate(list_label2indices):
#         ipc_c = ipc_per_class[c] if c < len(ipc_per_class) else ipc_per_class[-1]
#         idx_shuffle = np.random.permutation(indices)
#         labeled   = idx_shuffle[:ipc_c]
#         unlabeled = idx_shuffle[ipc_c:]
#         list_label2indices_labeled.append(labeled)
#         list_label2indices_unlabeled.append(unlabeled)
#     return list_label2indices_labeled, list_label2indices_unlabeled
# from Dataset.sample_dirichlet import clients_indices, clients_indices_homo

# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# CLASS_NAMES_HAM = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


# # ══════════════════════════════════════════════════════════════
# # 1. MODEL FACTORY
# # ══════════════════════════════════════════════════════════════

# def build_model(args):
#     if args.model == 'resnet8':
#         from Model.resnet8 import ResNet8
#         model = ResNet8(
#             resnet_size=8, scaling=4, save_activations=False,
#             group_norm_num_groups=args.group_norm_num_groups if args.use_groupnorm else None,
#             freeze_bn=False, freeze_bn_affine=False,
#             num_classes=args.num_classes,
#         )
#         if args.dataset == 'HAM10000':
#             model.avgpool = nn.AdaptiveAvgPool2d(1)

#     elif args.model == 'resnet18':
#         from torchvision.models import ResNet18_Weights
#         model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
#         model.fc = nn.Linear(model.fc.in_features, args.num_classes)
#         if args.use_groupnorm:
#             _convert_bn_to_gn(model, args.group_norm_num_groups)
#         else:
#             for m in model.modules():
#                 if isinstance(m, nn.BatchNorm2d):
#                     m.eval()
#                     m.weight.requires_grad = False
#                     m.bias.requires_grad = False
#     else:
#         raise ValueError(f"Bilinmeyen model: {args.model}")
#     return model


# def _convert_bn_to_gn(model, num_groups=16):
#     def _replace(module):
#         for name, child in module.named_children():
#             if isinstance(child, nn.BatchNorm2d):
#                 nc = child.num_features
#                 g = num_groups
#                 if nc % g != 0:
#                     for fg in [8, 4, 2, 1]:
#                         if nc % fg == 0:
#                             g = fg; break
#                 gn = nn.GroupNorm(g, nc, affine=True, eps=1e-5)
#                 if child.weight is not None:
#                     with torch.no_grad(): gn.weight.copy_(child.weight)
#                 if child.bias is not None:
#                     with torch.no_grad(): gn.bias.copy_(child.bias)
#                 setattr(module, name, gn)
#             else:
#                 _replace(child)
#     _replace(model)


# def forward_model(model, x):
#     out = model(x)
#     return out if isinstance(out, tuple) else (None, out)


# # ══════════════════════════════════════════════════════════════
# # 2. LOSS
# # ══════════════════════════════════════════════════════════════

# class FocalLoss(nn.Module):
#     def __init__(self, alpha_weights=None, gamma=2.0, reduction='mean'):
#         super().__init__()
#         self.alpha_weights = alpha_weights
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, logits, targets):
#         ce = F.cross_entropy(logits, targets, weight=self.alpha_weights, reduction='none')
#         pt = torch.exp(-ce)
#         focal = (1.0 - pt) ** self.gamma * ce
#         return focal.mean() if self.reduction == 'mean' else focal.sum()


# def build_criterion(class_counts, num_classes, device, use_focal):
#     if not use_focal:
#         return nn.CrossEntropyLoss().to(device)
#     total = sum(class_counts.values()) + 1e-8
#     weights = [total / (num_classes * class_counts.get(c, 1)) for c in range(num_classes)]
#     w = torch.tensor(weights, dtype=torch.float32).to(device)
#     w = w / w.mean()
#     return FocalLoss(alpha_weights=w, gamma=2.0)


# # ══════════════════════════════════════════════════════════════
# # 3. EXPERIMENT İSİMLENDİRME
# # ══════════════════════════════════════════════════════════════

# def get_exp_name(args):
#     flags = []
#     if args.use_focal_loss:         flags.append('FL')
#     if args.use_stfl:               flags.append('STFL')
#     if args.use_weighted_sampler:   flags.append('WS')
#     if args.use_medical_augment:    flags.append('MA')
#     if args.use_cosine_lr:          flags.append('CLR')
#     if args.use_groupnorm:          flags.append('GN')
#     if args.use_ema_eval:           flags.append('EMA')
#     if args.use_shapley_ema:        flags.append('SEMA')
#     if args.use_personalized_init:  flags.append('PINIT')
#     flag_str = '_'.join(flags) if flags else 'BASE'
#     return (
#         f"{args.dataset}_a{args.alpha}_{args.aggregation_method}_{args.model}_"
#         f"{flag_str}_L{args.num_labeled}_C{args.num_online_clients}_"
#         f"E{args.local_epochs}_T{args.threshold}_LR{args.lr_local_training}"
#     )


# # ══════════════════════════════════════════════════════════════
# # 4. CHECKPOINT
# # ══════════════════════════════════════════════════════════════

# def save_checkpoint(round_num, model_state, scheduler_state, metrics_history,
#                     local_ckpt_dir, args, filename='checkpoint.pt',
#                     backup_every=5, extra_state=None):
#     folder_name = get_exp_name(args)
#     os.makedirs(local_ckpt_dir, exist_ok=True)
#     state = {
#         'round': round_num,
#         'model_state_dict': model_state,
#         'scheduler_state_dict': scheduler_state,
#         'metrics_history': metrics_history,
#         'args': args,
#     }
#     if extra_state:
#         state.update(extra_state)
#     local_path = os.path.join(local_ckpt_dir, filename)
#     torch.save(state, local_path)
#     if round_num % backup_every == 0 or round_num == args.num_rounds:
#         try:
#             s3 = boto3.client('s3')
#             s3.upload_file(local_path, args.s3_bucket,
#                            f"checkpoints/{folder_name}/{filename}")
#             local_csv = f'./results/{args.dataset}/{get_exp_name(args)}.csv'
#             if os.path.exists(local_csv):
#                 s3.upload_file(local_csv, args.s3_bucket,
#                                f"results/{folder_name}/metrics.csv")
#             print(f"[S3] Round {round_num} yedeklendi.")
#         except Exception as e:
#             print(f"[S3-WARN] {e}")


# def load_checkpoint(model, scheduler, local_ckpt_dir, args,
#                     filename='checkpoint.pt', global_model_ref=None):
#     folder_name = get_exp_name(args)
#     local_path = os.path.join(local_ckpt_dir, filename)
#     s3_path = f"checkpoints/{folder_name}/{filename}"
#     if not os.path.exists(local_path):
#         try:
#             s3 = boto3.client('s3')
#             print(f"[CKPT] S3'ten indiriliyor: {s3_path}")
#             os.makedirs(local_ckpt_dir, exist_ok=True)
#             s3.download_file(args.s3_bucket, s3_path, local_path)
#         except ClientError:
#             print("[CKPT] Bulunamadi. Sifirdan basliyor.")
#             return 1, {'acc': [], 'acsa': [], 'f1': []}
#     print(f"[CKPT] Yukleniyor: {local_path}")
#     try:
#         ckpt = torch.load(local_path,
#                           map_location='cuda' if torch.cuda.is_available() else 'cpu',
#                           weights_only=False)
#         model.load_state_dict(ckpt['model_state_dict'])
#         if scheduler is not None and 'scheduler_state_dict' in ckpt:
#             scheduler.load_state_dict(ckpt['scheduler_state_dict'])
#         # EMA parametrelerini geri yukle
#         if global_model_ref is not None:
#             if 'ema_params' in ckpt and ckpt['ema_params'] is not None:
#                 global_model_ref.ema_params = ckpt['ema_params']
#                 print("[CKPT] EMA parametreleri yuklendi.")
#             if 'shapley_ema_store' in ckpt and ckpt['shapley_ema_store']:
#                 global_model_ref.shapley_ema_store = ckpt['shapley_ema_store']
#                 print("[CKPT] Shapley EMA store yuklendi.")
#             if 'client_gamma' in ckpt and ckpt['client_gamma']:
#                 global_model_ref.client_gamma = ckpt['client_gamma']
#                 print("[CKPT] Client gamma skorlari yuklendi.")
#         return ckpt['round'] + 1, ckpt['metrics_history']
#     except Exception as e:
#         print(f"[CKPT] Yukleme hatasi: {e}. Sifirdan basliyor.")
#         return 1, {'acc': [], 'acsa': [], 'f1': []}


# # ══════════════════════════════════════════════════════════════
# # 5. SHAPLEY (CSSV)
# # ══════════════════════════════════════════════════════════════

# def _get_classifier_keys(args):
#     if args.model == 'resnet18':
#         return 'fc.weight', 'fc.bias'
#     return 'classifier.weight', 'classifier.bias'


# def compute_cssv(args, local_models_params, initial_global_params):
#     num_clients = len(local_models_params)
#     num_classes = args.num_classes
#     if num_clients == 0:
#         return np.array([])

#     weight_layer, bias_layer = _get_classifier_keys(args)
#     # Tum hesaplamalari CPU'da yap (cihaz uyumsuzlugu engellenir)
#     g_weight = initial_global_params[weight_layer].float().cpu()
#     g_bias   = initial_global_params[bias_layer].float().cpu()
#     client_updates = []
#     for lp in local_models_params:
#         client_updates.append({
#             weight_layer: lp[weight_layer].float().cpu() - g_weight,
#             bias_layer:   lp[bias_layer].float().cpu()   - g_bias,
#         })

#     shapley_values = np.zeros((num_clients, num_classes))
#     num_samples = getattr(args, 'shapley_samples', 10)

#     for _ in range(num_samples):
#         perm = np.random.permutation(num_clients)
#         for i, cid in enumerate(perm):
#             coal   = perm[:i]
#             coal_p = perm[:i + 1]
#             for c in range(num_classes):
#                 cw = torch.cat([client_updates[cid][weight_layer][c].view(-1),
#                                 client_updates[cid][bias_layer][c].view(-1)])
#                 cn = F.normalize(cw.unsqueeze(0), p=2) if torch.norm(cw) > 0 else cw.unsqueeze(0)

#                 def _avg_agg(indices):
#                     if len(indices) == 0: return None
#                     tw = sum(client_updates[j][weight_layer][c] for j in indices) / len(indices)
#                     tb = sum(client_updates[j][bias_layer][c]   for j in indices) / len(indices)
#                     return torch.cat([tw.view(-1), tb.view(-1)])

#                 def _sim(agg):
#                     if agg is None or torch.norm(agg) == 0 or torch.norm(cw) == 0:
#                         return 0.0
#                     return F.cosine_similarity(cn, F.normalize(agg.unsqueeze(0), p=2)).item()

#                 shapley_values[cid, c] += _sim(_avg_agg(coal_p)) - _sim(_avg_agg(coal))

#     if num_samples > 0:
#         shapley_values /= num_samples

#     # Equation 6: normalize to [0,1] then compute gamma
#     # Γᵢ,ⱼ is in [-1,1], map to [0,1] via (1 + Γ) / 2
#     gamma = np.mean((1 + shapley_values) / 2, axis=1)   # shape [num_clients]

#     # Sütun normalizasyonu (aggregation icin gamma_tilde)
#     shapley_values = np.maximum(shapley_values, 0)
#     for c in range(num_classes):
#         s = shapley_values[:, c].sum()
#         shapley_values[:, c] = shapley_values[:, c] / s if s > 0 else np.ones(num_clients) / num_clients

#     return shapley_values, gamma


# # ══════════════════════════════════════════════════════════════
# # 6. GLOBAL MODEL
# # ══════════════════════════════════════════════════════════════

# class Global:
#     def __init__(self, args):
#         self.model = build_model(args)
#         self.model.cuda(args.gpu_id)
#         self.num_classes = args.num_classes
#         self.args = args

#         # V1: EMA eval state
#         self.ema_params = None
#         self.ema_decay  = getattr(args, 'ema_decay', 0.95)

#         # V2: Shapley EMA store — {str(client_id): np.array[num_classes]}
#         self.shapley_ema_store = {}
#         self.shapley_ema_decay = getattr(args, 'shapley_ema_decay', 0.7)

#         # V3 (Paper Algorithm 1): gamma skorlari — {str(client_id): float}
#         # gamma_i = (1/M) * sum((1 + Gamma_i,j) / 2)  ∈ [0,1]
#         # Baslangiçta 1/n ile initialize et (Algorithm 1, line 1)
#         self.client_gamma = {}     # normalize edilmemis gamma (broadcast icin)
#         self.client_gamma_tilde = {}  # normalize edilmis gamma (aggregation icin)

#     # ── V1: EMA Guncelleme ────────────────────────────────────
#     def update_ema(self, new_params):
#         """
#         global_model_ema = decay * prev_ema + (1-decay) * new_params
#         Tum tensorler CPU'da tutulur, cihaz uyumsuzlugu engellenir.
#         """
#         if self.ema_params is None:
#             self.ema_params = {k: v.float().cpu().clone() for k, v in new_params.items()}
#         else:
#             for k in self.ema_params:
#                 self.ema_params[k] = (
#                     self.ema_decay * self.ema_params[k].cpu() +
#                     (1.0 - self.ema_decay) * new_params[k].float().cpu()
#                 )

#     # ── Aggregation ──────────────────────────────────────────
#     def initialize_for_model_fusion(self, args, list_dicts_local_params,
#                                     list_nums_local_data, initial_global_params,
#                                     online_client_ids=None):
#         fused = copy.deepcopy(list_dicts_local_params[0])
#         nc    = len(list_dicts_local_params)
#         total = sum(list_nums_local_data)
#         weight_layer, bias_layer = _get_classifier_keys(args)

#         if args.aggregation_method == 'ShapFed':
#             cssv_raw, gamma_raw = compute_cssv(args, list_dicts_local_params, initial_global_params)

#             # V2: Shapley EMA yumusatma (cssv uzerinde)
#             if args.use_shapley_ema and online_client_ids is not None:
#                 cssv = np.zeros_like(cssv_raw)
#                 gamma = np.zeros_like(gamma_raw)
#                 for i, cid in enumerate(online_client_ids):
#                     key = str(cid)
#                     if key in self.shapley_ema_store:
#                         cssv[i] = (self.shapley_ema_decay * self.shapley_ema_store[key] +
#                                    (1 - self.shapley_ema_decay) * cssv_raw[i])
#                     else:
#                         cssv[i] = cssv_raw[i]
#                     self.shapley_ema_store[key] = cssv[i].copy()
#                 # gamma'yi EMA sonrasi cssv'den yeniden hesapla
#                 gamma = np.mean((1 + cssv) / 2, axis=1)
#                 # Sutun normalizasyonu yeniden uygula
#                 for c in range(args.num_classes):
#                     s = cssv[:, c].sum()
#                     cssv[:, c] = cssv[:, c] / s if s > 0 else np.ones(nc) / nc
#             else:
#                 cssv = cssv_raw
#                 gamma = gamma_raw

#             # Paper Equation 6: gamma_tilde = gamma / sum(gamma)
#             gamma_sum = gamma.sum()
#             gamma_tilde = gamma / gamma_sum if gamma_sum > 0 else np.ones(nc) / nc

#             # V3: Paper Algorithm 1, Line 6 - personalized broadcast icin gamma sakla
#             if args.use_personalized_init and online_client_ids is not None:
#                 for i, cid in enumerate(online_client_ids):
#                     key = str(cid)
#                     # Momentum ile yumusatma (paper: mu=0.9)
#                     mu = args.shapley_ema_decay if args.use_shapley_ema else 0.9
#                     if key in self.client_gamma:
#                         self.client_gamma[key] = mu * self.client_gamma[key] + (1 - mu) * float(gamma[i])
#                     else:
#                         self.client_gamma[key] = float(gamma[i])

#             # Aggregation icin gamma_tilde kullan (paper Equation 7)
#             cb_w = gamma_tilde
#         else:
#             cssv = None
#             gamma = None
#             cb_w = np.array([n / total for n in list_nums_local_data])

#         for name in list_dicts_local_params[0]:
#             orig_dtype = list_dicts_local_params[0][name].dtype
#             if args.aggregation_method == 'ShapFed' and name in (weight_layer, bias_layer):
#                 ft = torch.zeros_like(list_dicts_local_params[0][name], dtype=torch.float32)
#                 for c in range(args.num_classes):
#                     for i in range(nc):
#                         ft[c] += list_dicts_local_params[i][name][c] * cssv[i, c]
#             else:
#                 ft = sum(list_dicts_local_params[i][name].float() * float(cb_w[i])
#                          for i in range(nc))
#             fused[name] = ft.to(orig_dtype)
#         return fused

#     # ── Eval ─────────────────────────────────────────────────
#     def fedavg_eval(self, params, data_test, batch_size_test, args):
#         self.model.load_state_dict(params)
#         self.model.eval()
#         all_labels, all_preds = [], []
#         with torch.no_grad():
#             for imgs, labels in DataLoader(data_test, batch_size_test,
#                                            num_workers=0, pin_memory=False):
#                 imgs = imgs.cuda(args.gpu_id)
#                 _, logits = forward_model(self.model, imgs)
#                 _, preds = torch.max(logits, -1)
#                 all_labels.extend(labels.numpy())
#                 all_preds.extend(preds.cpu().numpy())

#         acc      = (np.array(all_labels) == np.array(all_preds)).mean()
#         acsa     = recall_score(all_labels, all_preds, average='macro', zero_division=0)
#         macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
#         pred_dist = dict(Counter(all_preds))

#         cm = confusion_matrix(all_labels, all_preds, labels=list(range(self.num_classes)))
#         per_class = {}
#         names = CLASS_NAMES_HAM if self.num_classes == 7 else [str(c) for c in range(self.num_classes)]
#         print("\n  +-- Per-Class Recall ----------------------")
#         for c in range(self.num_classes):
#             rs  = cm[c].sum()
#             rec = cm[c, c] / rs if rs > 0 else 0.0
#             per_class[names[c] if c < len(names) else str(c)] = round(rec, 4)
#             bar = '#' * int(rec * 20)
#             print(f"  | {names[c] if c < len(names) else str(c):<6}: {rec:.4f} {bar}")
#         print("  +------------------------------------------")

#         torch.cuda.empty_cache()
#         return acc, acsa, macro_f1, pred_dist, per_class

#     def download_params(self):
#         return self.model.state_dict()


# # ══════════════════════════════════════════════════════════════
# # 7. LOCAL MODEL
# # ══════════════════════════════════════════════════════════════

# class Local:
#     def __init__(self, args):
#         self.local_model = build_model(args)
#         self.local_G     = build_model(args)
#         self.local_model.cuda(args.gpu_id)
#         self.local_G.cuda(args.gpu_id)

#         self.optimizer = torch.optim.SGD(
#             self.local_model.parameters(),
#             lr=args.lr_local_training, momentum=0.9, weight_decay=1e-4,
#         )
#         if args.use_cosine_lr:
#             self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#                 self.optimizer, T_max=args.num_rounds,
#                 eta_min=args.lr_local_training * args.lr_min_ratio,
#             )
#         else:
#             self.scheduler = None

#         # V3: Client onceki parametreleri
#         # {str(client_id): state_dict(CPU)}
#         self.client_prev_params = {}

#     def get_init_params(self, client_id, global_params, args, global_model_ref):
#         """
#         Paper Algorithm 1, Line 6:
#             w̄ᵢ = γᵢ * wₛ + (1 − γᵢ) * wᵢ

#         gamma_i ∈ [0,1]:
#           - Yuksek gamma → daha cok global model al (iyi katkı)
#           - Dusuk gamma  → kendi parametresini koru (zayıf katkı)

#         Baslangiçta gamma = 1/n (saf global), client parametreleri
#         birikmesine gore personalization devreye girer.
#         """
#         if (not args.use_personalized_init or
#                 args.aggregation_method != 'ShapFed'):
#             return copy.deepcopy(global_params)

#         key = str(client_id)
#         n = args.num_clients

#         # gamma: paper Eq.6, momentum ile yumusatilmis
#         # Baslangiç: 1/n (Algorithm 1, Line 1)
#         gamma = global_model_ref.client_gamma.get(key, 1.0 / n)
#         # personalization_strength ile ölçekle (hyperparameter)
#         gamma = min(max(gamma * args.personalization_strength, 0.0), 1.0)

#         if key not in self.client_prev_params:
#             # Ilk round: client parametresi yok, saf global gonder
#             return copy.deepcopy(global_params)

#         # Paper Eq.8: w̄ᵢ = γᵢ * wₛ + (1 − γᵢ) * wᵢ
#         prev = self.client_prev_params[key]
#         personalized = {}
#         for k in global_params:
#             g_cpu = global_params[k].float().cpu()
#             p_cpu = prev[k].float().cpu()
#             personalized[k] = (gamma * g_cpu + (1.0 - gamma) * p_cpu).to(global_params[k].dtype)
#         return personalized

#     def store_client_params(self, client_id, local_params):
#         """Round sonunda client parametrelerini sakla."""
#         self.client_prev_params[str(client_id)] = {
#             k: v.cpu().clone() for k, v in local_params.items()
#         }

#     def fixmatch_train(self, args, data_labeled, data_unlabeled, init_params, r):
#         device = f'cuda:{args.gpu_id}'

#         local_labels = [int(data_labeled.dataset.targets[data_labeled.client_dataset[i][0]])
#                         for i in range(len(data_labeled))]
#         class_counts = dict(Counter(local_labels))

#         criterion = build_criterion(class_counts, args.num_classes, device,
#                                     use_focal=args.use_focal_loss)

#         if args.use_weighted_sampler and data_labeled.sample_weights:
#             sampler = WeightedRandomSampler(
#                 weights=torch.tensor(data_labeled.sample_weights, dtype=torch.float64),
#                 num_samples=len(data_labeled.sample_weights), replacement=True,
#             )
#             labeled_loader = DataLoader(data_labeled, sampler=sampler,
#                                         batch_size=args.batch_size_local_labeled_fixmatch,
#                                         drop_last=True, num_workers=0, pin_memory=False)
#         else:
#             labeled_loader = DataLoader(data_labeled, sampler=RandomSampler(data_labeled),
#                                         batch_size=args.batch_size_local_labeled_fixmatch,
#                                         drop_last=True, num_workers=0, pin_memory=False)

#         unlabeled_loader = DataLoader(data_unlabeled, sampler=RandomSampler(data_unlabeled),
#                                       batch_size=args.batch_size_local_labeled_fixmatch * args.mu,
#                                       drop_last=True, num_workers=0, pin_memory=False)

#         # init_params CPU'da tutulur, modele yuklemeden once GPU'ya tasipip
#         # load_state_dict map_location bunu handle eder ama explicit yapalim
#         self.local_model.load_state_dict(
#             {k: v.to(device) for k, v in init_params.items()})
#         self.local_model.train()
#         self.local_G.load_state_dict(
#             {k: v.to(device) for k, v in init_params.items()})
#         self.local_G.eval()

#         class_probs_ema = torch.ones(args.num_classes, device=device) / args.num_classes

#         MAX_ITER = 30
#         local_iter = max(min(int(data_unlabeled.client_dataset_len /
#                                args.batch_size_local_labeled_fixmatch), MAX_ITER), 5)

#         total_lx, total_lu, total_batches = 0.0, 0.0, 0
#         epoch_pseudo = []

#         for _ in range(args.local_epochs):
#             lab_iter   = iter(labeled_loader)
#             unlab_iter = iter(unlabeled_loader)

#             for _ in range(local_iter):
#                 try:
#                     inputs_x, targets_x = next(lab_iter)
#                 except StopIteration:
#                     lab_iter = iter(labeled_loader)
#                     inputs_x, targets_x = next(lab_iter)

#                 try:
#                     inputs_u_w, inputs_u_s, _ = next(unlab_iter)
#                 except StopIteration:
#                     unlab_iter = iter(unlabeled_loader)
#                     inputs_u_w, inputs_u_s, _ = next(unlab_iter)

#                 inputs_x   = inputs_x.to(device)
#                 inputs_u_w = inputs_u_w.to(device)
#                 inputs_u_s = inputs_u_s.to(device)
#                 targets_x  = targets_x.to(device)
#                 bs = inputs_x.shape[0]

#                 combined = self.interleave(
#                     torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1).to(device)
#                 _, logits = forward_model(self.local_model, combined)
#                 logits = self.de_interleave(logits, 2 * args.mu + 1)
#                 logits_x         = logits[:bs]
#                 logits_u_w_loc, logits_u_s = logits[bs:].chunk(2)
#                 del logits

#                 Lx = criterion(logits_x, targets_x)

#                 with torch.no_grad():
#                     _, logits_g = forward_model(self.local_G, inputs_u_w)
#                 pseudo_g = torch.softmax(logits_g / args.T, dim=-1)
#                 max_pg, target_g = torch.max(pseudo_g, dim=-1)

#                 pseudo_l = torch.softmax(logits_u_w_loc.detach() / args.T, dim=-1)
#                 max_pl, target_l = torch.max(pseudo_l, dim=-1)

#                 if args.use_stfl:
#                     class_probs_ema = class_probs_ema * 0.99 + pseudo_l.mean(0).detach() * 0.01
#                     ema_norm = class_probs_ema / (class_probs_ema.max() + 1e-8)
#                     beta = args.stfl_beta
#                     dyn_thresh = args.threshold * (1.0 + beta * ema_norm)
#                     dyn_thresh = torch.clamp(dyn_thresh, min=args.threshold, max=0.99)
#                     mask_l = max_pl.ge(dyn_thresh[target_l]).float()
#                     mask_g = max_pg.ge(dyn_thresh[target_g]).float()
#                 else:
#                     mask_l = max_pl.ge(args.threshold).float()
#                     mask_g = max_pg.ge(args.threshold).float()

#                 tl_oh = F.one_hot(target_l, args.num_classes).float()
#                 tg_oh = F.one_hot(target_g, args.num_classes).float()
#                 # SAGE CDSC Equation 3: lambda = exp(-kappa * delta_C)
#                 # Paper default kappa=0.5; dusuk delta -> lambda yakın 1 (local'e güven)
#                 #                          yüksek delta -> lambda yakın 0 (global'e güven)
#                 delta = torch.clamp(torch.abs(max_pl - max_pg) + 1e-6, 1e-6, 1.0)
#                 lam   = torch.clamp(
#                     torch.exp(-args.kappa * delta), 1e-6, 1.0)

#                 final_t = torch.where(
#                     mask_l.unsqueeze(1).bool(),
#                     lam.unsqueeze(1) * tl_oh + (1 - lam).unsqueeze(1) * tg_oh,
#                     tg_oh,
#                 )
#                 mask_valid = torch.max(mask_l, mask_g)

#                 logits_s_prob = torch.softmax(logits_u_s, dim=-1) + 1e-10
#                 Lu = (F.kl_div(logits_s_prob.log(), final_t + 1e-10, reduction='none')
#                       .sum(-1) * mask_valid).mean()

#                 loss = Lx + args.lambda_u * Lu
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), 10.0)
#                 self.optimizer.step()

#                 total_lx += Lx.item()
#                 total_lu += Lu.item()
#                 total_batches += 1
#                 epoch_pseudo.extend(target_l[mask_valid.bool()].cpu().numpy().tolist())

#         final_state = {k: v.cpu() for k, v in self.local_model.state_dict().items()}
#         self.optimizer.zero_grad(set_to_none=True)
#         torch.cuda.empty_cache()

#         return (final_state, dict(Counter(epoch_pseudo)),
#                 total_lx / max(total_batches, 1),
#                 total_lu / max(total_batches, 1))

#     def step_scheduler(self):
#         if self.scheduler is not None:
#             self.scheduler.step()

#     def get_lr(self):
#         return self.optimizer.param_groups[0]['lr']

#     @staticmethod
#     def interleave(x, size):
#         s = list(x.shape)
#         return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

#     @staticmethod
#     def de_interleave(x, size):
#         s = list(x.shape)
#         return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


# # ══════════════════════════════════════════════════════════════
# # 8. YARDIMCI
# # ══════════════════════════════════════════════════════════════

# class _SubsetImageFolder(torch.utils.data.Dataset):
#     def __init__(self, base_dataset, indices, transform=None):
#         self.base_dataset  = base_dataset
#         self.transform     = transform
#         self.valid_indices = [i for i in indices if i < len(base_dataset)]
#         self.targets       = [base_dataset.targets[i] for i in self.valid_indices]

#     def __len__(self): return len(self.valid_indices)

#     def __getitem__(self, idx):
#         try:
#             img, label = self.base_dataset[self.valid_indices[idx]]
#             if self.transform: img = self.transform(img)
#             return img, label
#         except Exception:
#             return self.__getitem__((idx + 1) % len(self.valid_indices))


# # ══════════════════════════════════════════════════════════════
# # 9. MAIN LOOP
# # ══════════════════════════════════════════════════════════════

# def main_loop(alpha):
#     args = args_parser()
#     args.alpha = alpha

#     exp_name       = get_exp_name(args)
#     local_ckpt_dir = os.path.join(args.checkpoint_dir, exp_name)

#     os.makedirs(f'./results/{args.dataset}/logs', exist_ok=True)
#     logging.basicConfig(
#         level=logging.INFO, format='%(asctime)s %(message)s',
#         filename=f'./results/{args.dataset}/logs/{exp_name}.log', filemode='a',
#     )

#     # ── Veri Seti ────────────────────────────────────────────
#     if args.dataset == 'HAM10000':
#         args.num_classes = 7
#         ham_mean, ham_std = [0.763, 0.545, 0.570], [0.140, 0.152, 0.169]
#         transform_test = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=ham_mean, std=ham_std),
#         ])
#         full = ImageFolder(root=args.path_ham10000, transform=None)
#         idx_tr, idx_te = train_test_split(range(len(full)), test_size=0.20,
#                                           stratify=full.targets, random_state=args.seed)
#         data_local_training = _SubsetImageFolder(full, idx_tr)
#         data_global_test    = _SubsetImageFolder(
#             ImageFolder(root=args.path_ham10000, transform=transform_test), idx_te)
#         print(f"[HAM10000] Train: {len(idx_tr)} | Test: {len(idx_te)}")
#         tc = Counter(full.targets[i] for i in idx_tr)
#         print(f"  Train dist: { {full.classes[k]: v for k, v in sorted(tc.items())} }")

#     elif args.dataset == 'CIFAR10':
#         args.num_classes = 10
#         transform_test = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
#         ])
#         data_local_training = datasets.CIFAR10(
#             args.path_cifar10, train=True, download=True, transform=None)
#         data_global_test    = datasets.CIFAR10(
#             args.path_cifar10, train=False, download=True, transform=transform_test)
#         print(f"[CIFAR10] Train: {len(data_local_training)} | Test: {len(data_global_test)}")
#     else:
#         raise ValueError(f"Desteklenmeyen dataset: {args.dataset}")

#     os.makedirs(local_ckpt_dir, exist_ok=True)

#     # ── Aktif Flagleri Yazdir ─────────────────────────────────
#     print(f"\n{'='*65}")
#     print(f"[EXP] {exp_name}")
#     print(f"{'='*65}")
#     print(f"  Model            : {args.model}")
#     print(f"  Aggregation      : {args.aggregation_method} (samples={args.shapley_samples})")
#     print(f"  --- Mevcut ---")
#     print(f"  Focal Loss       : {'ON' if args.use_focal_loss else 'off'}")
#     print(f"  STFL             : {'ON' if args.use_stfl else 'off'} (beta={args.stfl_beta})")
#     print(f"  Weighted Sampler : {'ON' if args.use_weighted_sampler else 'off'}")
#     print(f"  Medical Augment  : {'ON' if args.use_medical_augment else 'off'}")
#     print(f"  Cosine LR        : {'ON' if args.use_cosine_lr else 'off'}")
#     print(f"  GroupNorm        : {'ON' if args.use_groupnorm else 'off'}")
#     print(f"  --- Yeni ShapFed ---")
#     print(f"  EMA Eval         : {'ON' if args.use_ema_eval else 'off'} (decay={args.ema_decay})")
#     print(f"  Shapley EMA      : {'ON' if args.use_shapley_ema else 'off'} (decay={args.shapley_ema_decay})")
#     print(f"  Personalized Init: {'ON' if args.use_personalized_init else 'off'} (strength={args.personalization_strength})")
#     print(f"{'='*65}\n")

#     # ── Veri Dagitimy ────────────────────────────────────────
#     rng = np.random.RandomState(args.seed)
#     list_label2indices = classify_label(data_local_training, args.num_classes)

#     # IPC hesapla: num_labeled verilmisse sabit kullan,
#     # yoksa label_ratio ile her sinif icin ayri hesapla
#     if args.num_labeled is not None:
#         # Sabit IPC modu (eski davranis)
#         ipc = args.num_labeled
#         ipc_per_class = [ipc] * args.num_classes
#         print(f"[DATA] Sabit IPC={ipc} | Toplam etiketli~{ipc * args.num_classes}")
#     else:
#         # Oransal mod: her sinifin label_ratio kadar labeled
#         ipc_per_class = []
#         for c in range(args.num_classes):
#             n_class = len(list_label2indices[c])
#             n_labeled = max(1, int(n_class * args.label_ratio))
#             # Guvence: en az 1, en fazla sinif buyuklugu kadar
#             n_labeled = min(n_labeled, n_class)
#             ipc_per_class.append(n_labeled)
#         ipc = ipc_per_class  # partition_train listeli IPC alacak sekilde guncellendi
#         names_tmp = CLASS_NAMES_HAM if args.num_classes == 7 else [str(c) for c in range(args.num_classes)]
#         print(f"[DATA] Oransal IPC (ratio={args.label_ratio}):")
#         total_labeled = 0
#         for c, n in enumerate(ipc_per_class):
#             n_total = len(list_label2indices[c])
#             pct = n / n_total * 100
#             cname = names_tmp[c] if c < len(names_tmp) else str(c)
#             print(f"  {cname:<6}: {n:>4}/{n_total:<5} ({pct:.1f}% labeled)")
#             total_labeled += n
#         print(f"  Toplam labeled: {total_labeled}")

#     # partition_train: sabit IPC veya per-class IPC listesi destekler
#     if isinstance(ipc, list):
#         l_lab, l_unlab = partition_train_perclass(list_label2indices, ipc)
#     else:
#         l_lab, l_unlab = partition_train(list_label2indices, ipc)
#     if alpha == 0:
#         c_lab   = clients_indices_homo(l_lab,   args.num_classes, args.num_clients)
#         c_unlab = clients_indices_homo(l_unlab, args.num_classes, args.num_clients)
#     else:
#         c_lab   = clients_indices(l_lab,   args.num_classes, args.num_clients, alpha, seed=0)
#         c_unlab = clients_indices(l_unlab, args.num_classes, args.num_clients, alpha, seed=0)

#     for i in range(len(c_lab)):
#         if i < len(c_unlab): c_unlab[i].extend(c_lab[i])
#         else: c_unlab.append(list(c_lab[i]))

#     # ── Modeller ─────────────────────────────────────────────
#     global_model = Global(args)
#     local_model  = Local(args)

#     start_round, metrics_history = load_checkpoint(
#         global_model.model, local_model.scheduler, local_ckpt_dir, args,
#         global_model_ref=global_model)

#     idx_labeled   = Indices2Dataset_labeled(
#         data_local_training, dataset_name=args.dataset,
#         use_medical_augment=args.use_medical_augment)
#     idx_unlabeled = Indices2Dataset_unlabeled_fixmatch(
#         data_local_training, dataset_name=args.dataset,
#         use_medical_augment=args.use_medical_augment)

#     dashboard_data = {}
#     os.makedirs(f'./results/{args.dataset}', exist_ok=True)
#     total_clients = list(range(args.num_clients))
#     names = CLASS_NAMES_HAM if args.num_classes == 7 else [str(c) for c in range(args.num_classes)]

#     print(f"[TRAIN] Round {start_round} -> {args.num_rounds}")

#     for r in tqdm(range(start_round, args.num_rounds + 1), desc='Server'):
#         dict_global_params = global_model.download_params()
#         online = rng.choice(total_clients, args.num_online_clients, replace=False)

#         list_params, list_nums = [], []
#         round_dists, round_pseudo = {}, Counter()

#         for client in online:
#             idx_labeled.load(c_lab[client])
#             idx_unlabeled.load(c_unlab[client])

#             lbl_counts = Counter(data_local_training.targets[i]
#                                  for i in c_lab[client] if i < len(data_local_training))
#             round_dists[str(client)] = {str(k): v for k, v in lbl_counts.items()}

#             summary = ', '.join(
#                 f"{names[int(k)] if int(k) < len(names) else k}: {v}"
#                 for k, v in sorted(lbl_counts.items()))
#             print(f"\n> Client {client}  [LR={local_model.get_lr():.6f}]")
#             print(f"    Labeled  : [{summary}]")
#             print(f"    Unlabeled: {len(c_unlab[client])} adet")

#             list_nums.append(len(c_lab[client]) + len(c_unlab[client]))

#             # V3: Personalized init parametrelerini al
#             init_params = local_model.get_init_params(
#                 client, copy.deepcopy(dict_global_params), args, global_model)

#             params, pseudo, lx, lu = local_model.fixmatch_train(
#                 args, idx_labeled, idx_unlabeled, init_params, r)

#             # V3: Client parametrelerini sakla (sonraki round icin)
#             if args.use_personalized_init:
#                 local_model.store_client_params(client, params)

#             list_params.append(copy.deepcopy(params))
#             round_pseudo.update(pseudo)

#             p_summary = ', '.join(
#                 f"{names[int(k)] if int(k) < len(names) else k}: {v}"
#                 for k, v in sorted(pseudo.items()))
#             print(f"    Lx: {lx:.4f} | Lu: {lu:.4f}")
#             print(f"    Pseudo: [{p_summary}]")
#             del params
#             torch.cuda.empty_cache()

#         # Aggregation
#         fedavg_params = global_model.initialize_for_model_fusion(
#             args, list_params, list_nums, dict_global_params,
#             online_client_ids=list(online))
#         global_model.model.load_state_dict(fedavg_params)

#         # V1: EMA guncelle
#         global_model.update_ema(fedavg_params)

#         # Eval: EMA aktifse ema_params kullan
#         if args.use_ema_eval and global_model.ema_params is not None:
#             eval_params = {k: v.to(fedavg_params[k].dtype)
#                            for k, v in global_model.ema_params.items()}
#         else:
#             eval_params = fedavg_params

#         acc, acsa, f1, pred_dist, per_class = global_model.fedavg_eval(
#             copy.deepcopy(eval_params), data_global_test, args.batch_size_test, args)

#         metrics_history['acc'].append(acc)
#         metrics_history['acsa'].append(acsa)
#         metrics_history['f1'].append(f1)

#         b_acc  = max(metrics_history['acc'])
#         b_acsa = max(metrics_history['acsa'])
#         b_f1   = max(metrics_history['f1'])

#         print(f"\n[Round {r:>4}/{args.num_rounds}] "
#               f"Acc:{acc:.4f} ACSA:{acsa:.4f} F1:{f1:.4f} || "
#               f"Best-> Acc:{b_acc:.4f} ACSA:{b_acsa:.4f} F1:{b_f1:.4f}")
#         logging.info(f"Round {r} | Acc:{acc:.4f} ACSA:{acsa:.4f} F1:{f1:.4f} | "
#                      f"Best ACSA:{b_acsa:.4f} | LR:{local_model.get_lr():.6f} | {per_class}")

#         # Dashboard
#         dashboard_data[str(r)] = {
#             'client_distributions': round_dists,
#             'pseudo_labels': {str(k): v for k, v in round_pseudo.items()},
#             'global_predictions': {str(k): v for k, v in pred_dist.items()},
#             'per_class_recall': per_class,
#             'lr': local_model.get_lr(),
#         }
#         log_path = f'./results/{args.dataset}/dashboard_data.json'
#         with open(log_path, 'w') as f:
#             json.dump(dashboard_data, f, indent=2)
#         try:
#             boto3.client('s3').upload_file(
#                 log_path, args.s3_bucket, f"{exp_name}/dashboard_data.json")
#         except Exception as e:
#             print(f"[S3-WARN] {e}")

#         # Checkpoint — yeni state'leri de kaydet
#         extra = {
#             'ema_params':           global_model.ema_params,
#             'shapley_ema_store':    global_model.shapley_ema_store,
#             'client_gamma': global_model.client_gamma,
#         }
#         save_checkpoint(
#             r, global_model.download_params(),
#             local_model.scheduler.state_dict() if local_model.scheduler else {},
#             metrics_history, local_ckpt_dir, args, extra_state=extra)

#         # CSV
#         n = len(metrics_history['acc'])
#         pd.DataFrame({
#             'round': range(1, n + 1),
#             'acc':   metrics_history['acc'],
#             'acsa':  metrics_history['acsa'],
#             'f1':    metrics_history['f1'],
#         }).to_csv(f'./results/{args.dataset}/{exp_name}.csv', index=False)

#         local_model.step_scheduler()


# # ══════════════════════════════════════════════════════════════
# # 10. ENTRY POINT
# # ══════════════════════════════════════════════════════════════

# if __name__ == '__main__':
#     torch.manual_seed(7)
#     torch.cuda.manual_seed(7)
#     np.random.seed(7)
#     random.seed(7)
#     torch.backends.cudnn.deterministic = True
#     args = args_parser()
#     main_loop(args.alpha)

"""
SAGE_final.py — Unified Federated Semi-Supervised Learning
=============================================================

Tüm geliştirmeler tek dosyada, flag ile açılıp kapatılır.

Checkpoint isimlendirme:
  {dataset}_a{alpha}_{agg}_{model}_{flags}_L{ipc}_C{online}_E{epochs}_T{thr}_LR{lr}
  Flags: FL STFL WS MA CLR GN EMA SEMA PINIT

Örnek run komutları → run_commands.sh dosyasına bakın.
"""

import os, copy, random, logging, json
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
from collections import Counter
import torchvision.models as models

from options import args_parser
from Dataset.dataset import (
    classify_label, Indices2Dataset_labeled,
    Indices2Dataset_unlabeled_fixmatch, partition_train,
)

def partition_train_perclass(list_label2indices, ipc_per_class):
    """
    Her sinif icin farkli IPC degeri kullanan partition_train.
    ipc_per_class: [ipc_0, ipc_1, ..., ipc_C] listesi
    """
    list_label2indices_labeled   = []
    list_label2indices_unlabeled = []
    for c, indices in enumerate(list_label2indices):
        ipc_c = ipc_per_class[c] if c < len(ipc_per_class) else ipc_per_class[-1]
        idx_shuffle = np.random.permutation(indices)
        labeled   = idx_shuffle[:ipc_c]
        unlabeled = idx_shuffle[ipc_c:]
        list_label2indices_labeled.append(labeled)
        list_label2indices_unlabeled.append(unlabeled)
    return list_label2indices_labeled, list_label2indices_unlabeled
from Dataset.sample_dirichlet import clients_indices, clients_indices_homo

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

CLASS_NAMES_HAM = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


# ══════════════════════════════════════════════════════════════
# 1. MODEL FACTORY
# ══════════════════════════════════════════════════════════════

def build_model(args):
    if args.model == 'resnet8':
        from Model.resnet8 import ResNet8
        model = ResNet8(
            resnet_size=8, scaling=4, save_activations=False,
            group_norm_num_groups=args.group_norm_num_groups if args.use_groupnorm else None,
            freeze_bn=False, freeze_bn_affine=False,
            num_classes=args.num_classes,
        )
        if args.dataset == 'HAM10000':
            model.avgpool = nn.AdaptiveAvgPool2d(1)

    elif args.model == 'resnet18':
        from torchvision.models import ResNet18_Weights
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
        if args.use_groupnorm:
            _convert_bn_to_gn(model, args.group_norm_num_groups)
        else:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
    else:
        raise ValueError(f"Bilinmeyen model: {args.model}")
    return model


def _convert_bn_to_gn(model, num_groups=16):
    def _replace(module):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                nc = child.num_features
                g = num_groups
                if nc % g != 0:
                    for fg in [8, 4, 2, 1]:
                        if nc % fg == 0:
                            g = fg; break
                gn = nn.GroupNorm(g, nc, affine=True, eps=1e-5)
                if child.weight is not None:
                    with torch.no_grad(): gn.weight.copy_(child.weight)
                if child.bias is not None:
                    with torch.no_grad(): gn.bias.copy_(child.bias)
                setattr(module, name, gn)
            else:
                _replace(child)
    _replace(model)


def forward_model(model, x):
    out = model(x)
    return out if isinstance(out, tuple) else (None, out)


# ══════════════════════════════════════════════════════════════
# 2. LOSS
# ══════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, alpha_weights=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha_weights = alpha_weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha_weights, reduction='none')
        pt = torch.exp(-ce)
        focal = (1.0 - pt) ** self.gamma * ce
        return focal.mean() if self.reduction == 'mean' else focal.sum()


def build_criterion(class_counts, num_classes, device, use_focal):
    if not use_focal:
        return nn.CrossEntropyLoss().to(device)
    total = sum(class_counts.values()) + 1e-8
    weights = [total / (num_classes * class_counts.get(c, 1)) for c in range(num_classes)]
    w = torch.tensor(weights, dtype=torch.float32).to(device)
    w = w / w.mean()
    return FocalLoss(alpha_weights=w, gamma=2.0)


# ══════════════════════════════════════════════════════════════
# 3. EXPERIMENT İSİMLENDİRME
# ══════════════════════════════════════════════════════════════

def get_exp_name(args):
    flags = []
    if args.use_focal_loss:         flags.append('FL')
    if args.use_stfl:               flags.append('STFL')
    if args.use_weighted_sampler:   flags.append('WS')
    if args.use_medical_augment:    flags.append('MA')
    if args.use_cosine_lr:          flags.append('CLR')
    if args.use_groupnorm:          flags.append('GN')
    if args.use_ema_eval:           flags.append('EMA')
    if args.use_shapley_ema:        flags.append('SEMA')
    if args.use_personalized_init:  flags.append('PINIT')
    flag_str = '_'.join(flags) if flags else 'BASE'
    return (
        f"{args.dataset}_a{args.alpha}_{args.aggregation_method}_{args.model}_"
        f"{flag_str}_L{args.num_labeled}_C{args.num_online_clients}_"
        f"E{args.local_epochs}_T{args.threshold}_LR{args.lr_local_training}"
    )


# ══════════════════════════════════════════════════════════════
# 4. CHECKPOINT
# ══════════════════════════════════════════════════════════════

def save_checkpoint(round_num, model_state, scheduler_state, metrics_history,
                    local_ckpt_dir, args, filename='checkpoint.pt',
                    backup_every=5, extra_state=None):
    folder_name = get_exp_name(args)
    os.makedirs(local_ckpt_dir, exist_ok=True)
    state = {
        'round': round_num,
        'model_state_dict': model_state,
        'scheduler_state_dict': scheduler_state,
        'metrics_history': metrics_history,
        'args': args,
    }
    if extra_state:
        state.update(extra_state)
    local_path = os.path.join(local_ckpt_dir, filename)
    torch.save(state, local_path)
    if round_num % backup_every == 0 or round_num == args.num_rounds:
        try:
            s3 = boto3.client('s3')
            s3.upload_file(local_path, args.s3_bucket,
                           f"checkpoints/{folder_name}/{filename}")
            local_csv = f'./results/{args.dataset}/{get_exp_name(args)}.csv'
            if os.path.exists(local_csv):
                s3.upload_file(local_csv, args.s3_bucket,
                               f"results/{folder_name}/metrics.csv")
            print(f"[S3] Round {round_num} yedeklendi.")
        except Exception as e:
            print(f"[S3-WARN] {e}")


def load_checkpoint(model, scheduler, local_ckpt_dir, args,
                    filename='checkpoint.pt', global_model_ref=None):
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
            print("[CKPT] Bulunamadi. Sifirdan basliyor.")
            return 1, {'acc': [], 'acsa': [], 'f1': []}
    print(f"[CKPT] Yukleniyor: {local_path}")
    try:
        ckpt = torch.load(local_path,
                          map_location='cuda' if torch.cuda.is_available() else 'cpu',
                          weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        # EMA parametrelerini geri yukle
        if global_model_ref is not None:
            if 'ema_params' in ckpt and ckpt['ema_params'] is not None:
                global_model_ref.ema_params = ckpt['ema_params']
                print("[CKPT] EMA parametreleri yuklendi.")
            if 'shapley_ema_store' in ckpt and ckpt['shapley_ema_store']:
                global_model_ref.shapley_ema_store = ckpt['shapley_ema_store']
                print("[CKPT] Shapley EMA store yuklendi.")
            if 'client_gamma' in ckpt and ckpt['client_gamma']:
                global_model_ref.client_gamma = ckpt['client_gamma']
                print("[CKPT] Client gamma skorlari yuklendi.")
        return ckpt['round'] + 1, ckpt['metrics_history']
    except Exception as e:
        print(f"[CKPT] Yukleme hatasi: {e}. Sifirdan basliyor.")
        return 1, {'acc': [], 'acsa': [], 'f1': []}


# ══════════════════════════════════════════════════════════════
# 5. SHAPLEY (CSSV)
# ══════════════════════════════════════════════════════════════

def _get_classifier_keys(args):
    if args.model == 'resnet18':
        return 'fc.weight', 'fc.bias'
    return 'classifier.weight', 'classifier.bias'


def compute_cssv(args, local_models_params, initial_global_params):
    num_clients = len(local_models_params)
    num_classes = args.num_classes
    if num_clients == 0:
        return np.array([])

    weight_layer, bias_layer = _get_classifier_keys(args)
    # Tum hesaplamalari CPU'da yap (cihaz uyumsuzlugu engellenir)
    g_weight = initial_global_params[weight_layer].float().cpu()
    g_bias   = initial_global_params[bias_layer].float().cpu()
    client_updates = []
    for lp in local_models_params:
        client_updates.append({
            weight_layer: lp[weight_layer].float().cpu() - g_weight,
            bias_layer:   lp[bias_layer].float().cpu()   - g_bias,
        })

    shapley_values = np.zeros((num_clients, num_classes))
    num_samples = getattr(args, 'shapley_samples', 10)

    for _ in range(num_samples):
        perm = np.random.permutation(num_clients)
        for i, cid in enumerate(perm):
            coal   = perm[:i]
            coal_p = perm[:i + 1]
            for c in range(num_classes):
                cw = torch.cat([client_updates[cid][weight_layer][c].view(-1),
                                client_updates[cid][bias_layer][c].view(-1)])
                cn = F.normalize(cw.unsqueeze(0), p=2) if torch.norm(cw) > 0 else cw.unsqueeze(0)

                def _avg_agg(indices):
                    if len(indices) == 0: return None
                    tw = sum(client_updates[j][weight_layer][c] for j in indices) / len(indices)
                    tb = sum(client_updates[j][bias_layer][c]   for j in indices) / len(indices)
                    return torch.cat([tw.view(-1), tb.view(-1)])

                def _sim(agg):
                    if agg is None or torch.norm(agg) == 0 or torch.norm(cw) == 0:
                        return 0.0
                    return F.cosine_similarity(cn, F.normalize(agg.unsqueeze(0), p=2)).item()

                shapley_values[cid, c] += _sim(_avg_agg(coal_p)) - _sim(_avg_agg(coal))

    if num_samples > 0:
        shapley_values /= num_samples

    # Equation 6: normalize to [0,1] then compute gamma
    # Γᵢ,ⱼ is in [-1,1], map to [0,1] via (1 + Γ) / 2
    gamma = np.mean((1 + shapley_values) / 2, axis=1)   # shape [num_clients]

    # Sütun normalizasyonu (aggregation icin gamma_tilde)
    shapley_values = np.maximum(shapley_values, 0)
    for c in range(num_classes):
        s = shapley_values[:, c].sum()
        shapley_values[:, c] = shapley_values[:, c] / s if s > 0 else np.ones(num_clients) / num_clients

    return shapley_values, gamma


# ══════════════════════════════════════════════════════════════
# 6. GLOBAL MODEL
# ══════════════════════════════════════════════════════════════

class Global:
    def __init__(self, args):
        self.model = build_model(args)
        self.model.cuda(args.gpu_id)
        self.num_classes = args.num_classes
        self.args = args

        # V1: EMA eval state
        self.ema_params = None
        self.ema_decay  = getattr(args, 'ema_decay', 0.95)

        # V2: Shapley EMA store — {str(client_id): np.array[num_classes]}
        self.shapley_ema_store = {}
        self.shapley_ema_decay = getattr(args, 'shapley_ema_decay', 0.7)

        # V3 (Paper Algorithm 1): gamma skorlari — {str(client_id): float}
        # gamma_i = (1/M) * sum((1 + Gamma_i,j) / 2)  ∈ [0,1]
        # Baslangiçta 1/n ile initialize et (Algorithm 1, line 1)
        self.client_gamma = {}     # normalize edilmemis gamma (broadcast icin)
        self.client_gamma_tilde = {}  # normalize edilmis gamma (aggregation icin)

    # ── V1: EMA Guncelleme ────────────────────────────────────
    def update_ema(self, new_params):
        """
        global_model_ema = decay * prev_ema + (1-decay) * new_params
        Tum tensorler CPU'da tutulur, cihaz uyumsuzlugu engellenir.
        """
        if self.ema_params is None:
            self.ema_params = {k: v.float().cpu().clone() for k, v in new_params.items()}
        else:
            for k in self.ema_params:
                self.ema_params[k] = (
                    self.ema_decay * self.ema_params[k].cpu() +
                    (1.0 - self.ema_decay) * new_params[k].float().cpu()
                )

    # ── Aggregation ──────────────────────────────────────────
    def initialize_for_model_fusion(self, args, list_dicts_local_params,
                                    list_nums_local_data, initial_global_params,
                                    online_client_ids=None):
        fused = copy.deepcopy(list_dicts_local_params[0])
        nc    = len(list_dicts_local_params)
        total = sum(list_nums_local_data)
        weight_layer, bias_layer = _get_classifier_keys(args)

        if args.aggregation_method == 'ShapFed':
            cssv_raw, gamma_raw = compute_cssv(args, list_dicts_local_params, initial_global_params)

            # Paper Algorithm 1 Line 17: Γ = μΓ + (1-μ)Γ̃
            # momentum her zaman aktif (mu=0.9 paper default)
            # use_shapley_ema=True ise args.shapley_ema_decay kullan,
            # False ise paper default mu=0.9 kullan
            mu = args.shapley_ema_decay if args.use_shapley_ema else 0.9
            cssv = np.zeros_like(cssv_raw)
            if online_client_ids is not None:
                for i, cid in enumerate(online_client_ids):
                    key = str(cid)
                    if key in self.shapley_ema_store:
                        cssv[i] = mu * self.shapley_ema_store[key] + (1 - mu) * cssv_raw[i]
                    else:
                        cssv[i] = cssv_raw[i]
                    self.shapley_ema_store[key] = cssv[i].copy()
            else:
                cssv = cssv_raw
            # gamma'yi momentum sonrasi cssv'den hesapla
            gamma = np.mean((1 + cssv) / 2, axis=1)
            # Sutun normalizasyonu
            for c in range(args.num_classes):
                s = cssv[:, c].sum()
                cssv[:, c] = cssv[:, c] / s if s > 0 else np.ones(nc) / nc

            # Paper Equation 6: gamma_tilde = gamma / sum(gamma)
            gamma_sum = gamma.sum()
            gamma_tilde = gamma / gamma_sum if gamma_sum > 0 else np.ones(nc) / nc

            # V3: Paper Algorithm 1, Line 6 - personalized broadcast icin gamma sakla
            if args.use_personalized_init and online_client_ids is not None:
                for i, cid in enumerate(online_client_ids):
                    key = str(cid)
                    # Momentum ile yumusatma (paper: mu=0.9)
                    mu = args.shapley_ema_decay if args.use_shapley_ema else 0.9
                    if key in self.client_gamma:
                        self.client_gamma[key] = mu * self.client_gamma[key] + (1 - mu) * float(gamma[i])
                    else:
                        self.client_gamma[key] = float(gamma[i])

            # Aggregation icin gamma_tilde kullan (paper Equation 7)
            cb_w = gamma_tilde
        else:
            cssv = None
            gamma = None
            cb_w = np.array([n / total for n in list_nums_local_data])

        for name in list_dicts_local_params[0]:
            orig_dtype = list_dicts_local_params[0][name].dtype
            if args.aggregation_method == 'ShapFed' and name in (weight_layer, bias_layer):
                ft = torch.zeros_like(list_dicts_local_params[0][name], dtype=torch.float32)
                for c in range(args.num_classes):
                    for i in range(nc):
                        ft[c] += list_dicts_local_params[i][name][c] * cssv[i, c]
            else:
                ft = sum(list_dicts_local_params[i][name].float() * float(cb_w[i])
                         for i in range(nc))
            fused[name] = ft.to(orig_dtype)
        return fused

    # ── Eval ─────────────────────────────────────────────────
    def fedavg_eval(self, params, data_test, batch_size_test, args):
        self.model.load_state_dict(params)
        self.model.eval()
        all_labels, all_preds = [], []
        with torch.no_grad():
            for imgs, labels in DataLoader(data_test, batch_size_test,
                                           num_workers=0, pin_memory=False):
                imgs = imgs.cuda(args.gpu_id)
                _, logits = forward_model(self.model, imgs)
                _, preds = torch.max(logits, -1)
                all_labels.extend(labels.numpy())
                all_preds.extend(preds.cpu().numpy())

        acc      = (np.array(all_labels) == np.array(all_preds)).mean()
        acsa     = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        pred_dist = dict(Counter(all_preds))

        cm = confusion_matrix(all_labels, all_preds, labels=list(range(self.num_classes)))
        per_class = {}
        names = CLASS_NAMES_HAM if self.num_classes == 7 else [str(c) for c in range(self.num_classes)]
        print("\n  +-- Per-Class Recall ----------------------")
        for c in range(self.num_classes):
            rs  = cm[c].sum()
            rec = cm[c, c] / rs if rs > 0 else 0.0
            per_class[names[c] if c < len(names) else str(c)] = round(rec, 4)
            bar = '#' * int(rec * 20)
            print(f"  | {names[c] if c < len(names) else str(c):<6}: {rec:.4f} {bar}")
        print("  +------------------------------------------")

        torch.cuda.empty_cache()
        return acc, acsa, macro_f1, pred_dist, per_class

    def download_params(self):
        return self.model.state_dict()


# ══════════════════════════════════════════════════════════════
# 7. LOCAL MODEL
# ══════════════════════════════════════════════════════════════

class Local:
    def __init__(self, args):
        self.local_model = build_model(args)
        self.local_G     = build_model(args)
        self.local_model.cuda(args.gpu_id)
        self.local_G.cuda(args.gpu_id)

        self.optimizer = torch.optim.SGD(
            self.local_model.parameters(),
            lr=args.lr_local_training, momentum=0.9, weight_decay=1e-4,
        )
        if args.use_cosine_lr:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=args.num_rounds,
                eta_min=args.lr_local_training * args.lr_min_ratio,
            )
        else:
            self.scheduler = None

        # V3: Client onceki parametreleri
        # {str(client_id): state_dict(CPU)}
        self.client_prev_params = {}

    def get_init_params(self, client_id, global_params, args, global_model_ref):
        """
        Paper Algorithm 1, Line 6:
            w̄ᵢ = γᵢ * wₛ + (1 − γᵢ) * wᵢ

        gamma_i ∈ [0,1]:
          - Yuksek gamma → daha cok global model al (iyi katkı)
          - Dusuk gamma  → kendi parametresini koru (zayıf katkı)

        Baslangiçta gamma = 1/n (saf global), client parametreleri
        birikmesine gore personalization devreye girer.
        """
        if (not args.use_personalized_init or
                args.aggregation_method != 'ShapFed'):
            return copy.deepcopy(global_params)

        key = str(client_id)
        n = args.num_clients

        # gamma: paper Eq.6, momentum ile yumusatilmis
        # Baslangiç: 1/n (Algorithm 1, Line 1)
        gamma = global_model_ref.client_gamma.get(key, 1.0 / n)
        # personalization_strength ile ölçekle (hyperparameter)
        gamma = min(max(gamma * args.personalization_strength, 0.0), 1.0)

        if key not in self.client_prev_params:
            # Ilk round: client parametresi yok, saf global gonder
            return copy.deepcopy(global_params)

        # Paper Eq.8: w̄ᵢ = γᵢ * wₛ + (1 − γᵢ) * wᵢ
        prev = self.client_prev_params[key]
        personalized = {}
        for k in global_params:
            g_cpu = global_params[k].float().cpu()
            p_cpu = prev[k].float().cpu()
            personalized[k] = (gamma * g_cpu + (1.0 - gamma) * p_cpu).to(global_params[k].dtype)
        return personalized

    def store_client_params(self, client_id, local_params):
        """Round sonunda client parametrelerini sakla."""
        self.client_prev_params[str(client_id)] = {
            k: v.cpu().clone() for k, v in local_params.items()
        }

    def fixmatch_train(self, args, data_labeled, data_unlabeled, init_params, r):
        device = f'cuda:{args.gpu_id}'

        local_labels = [int(data_labeled.dataset.targets[data_labeled.client_dataset[i][0]])
                        for i in range(len(data_labeled))]
        class_counts = dict(Counter(local_labels))

        criterion = build_criterion(class_counts, args.num_classes, device,
                                    use_focal=args.use_focal_loss)

        if args.use_weighted_sampler and data_labeled.sample_weights:
            sampler = WeightedRandomSampler(
                weights=torch.tensor(data_labeled.sample_weights, dtype=torch.float64),
                num_samples=len(data_labeled.sample_weights), replacement=True,
            )
            labeled_loader = DataLoader(data_labeled, sampler=sampler,
                                        batch_size=args.batch_size_local_labeled_fixmatch,
                                        drop_last=True, num_workers=0, pin_memory=False)
        else:
            labeled_loader = DataLoader(data_labeled, sampler=RandomSampler(data_labeled),
                                        batch_size=args.batch_size_local_labeled_fixmatch,
                                        drop_last=True, num_workers=0, pin_memory=False)

        unlabeled_loader = DataLoader(data_unlabeled, sampler=RandomSampler(data_unlabeled),
                                      batch_size=args.batch_size_local_labeled_fixmatch * args.mu,
                                      drop_last=True, num_workers=0, pin_memory=False)

        # init_params CPU'da tutulur, modele yuklemeden once GPU'ya tasipip
        # load_state_dict map_location bunu handle eder ama explicit yapalim
        self.local_model.load_state_dict(
            {k: v.to(device) for k, v in init_params.items()})
        self.local_model.train()
        self.local_G.load_state_dict(
            {k: v.to(device) for k, v in init_params.items()})
        self.local_G.eval()

        class_probs_ema = torch.ones(args.num_classes, device=device) / args.num_classes

        MAX_ITER = 30
        local_iter = max(min(int(data_unlabeled.client_dataset_len /
                               args.batch_size_local_labeled_fixmatch), MAX_ITER), 5)

        total_lx, total_lu, total_batches = 0.0, 0.0, 0
        epoch_pseudo = []

        for _ in range(args.local_epochs):
            lab_iter   = iter(labeled_loader)
            unlab_iter = iter(unlabeled_loader)

            for _ in range(local_iter):
                try:
                    inputs_x, targets_x = next(lab_iter)
                except StopIteration:
                    lab_iter = iter(labeled_loader)
                    inputs_x, targets_x = next(lab_iter)

                try:
                    inputs_u_w, inputs_u_s, _ = next(unlab_iter)
                except StopIteration:
                    unlab_iter = iter(unlabeled_loader)
                    inputs_u_w, inputs_u_s, _ = next(unlab_iter)

                inputs_x   = inputs_x.to(device)
                inputs_u_w = inputs_u_w.to(device)
                inputs_u_s = inputs_u_s.to(device)
                targets_x  = targets_x.to(device)
                bs = inputs_x.shape[0]

                combined = self.interleave(
                    torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1).to(device)
                _, logits = forward_model(self.local_model, combined)
                logits = self.de_interleave(logits, 2 * args.mu + 1)
                logits_x         = logits[:bs]
                logits_u_w_loc, logits_u_s = logits[bs:].chunk(2)
                del logits

                Lx = criterion(logits_x, targets_x)

                with torch.no_grad():
                    _, logits_g = forward_model(self.local_G, inputs_u_w)
                pseudo_g = torch.softmax(logits_g / args.T, dim=-1)
                max_pg, target_g = torch.max(pseudo_g, dim=-1)

                pseudo_l = torch.softmax(logits_u_w_loc.detach() / args.T, dim=-1)
                max_pl, target_l = torch.max(pseudo_l, dim=-1)

                if args.use_stfl:
                    class_probs_ema = class_probs_ema * 0.99 + pseudo_l.mean(0).detach() * 0.01
                    ema_norm = class_probs_ema / (class_probs_ema.max() + 1e-8)
                    beta = args.stfl_beta
                    dyn_thresh = args.threshold * (1.0 + beta * ema_norm)
                    dyn_thresh = torch.clamp(dyn_thresh, min=args.threshold, max=0.99)
                    mask_l = max_pl.ge(dyn_thresh[target_l]).float()
                    mask_g = max_pg.ge(dyn_thresh[target_g]).float()
                else:
                    mask_l = max_pl.ge(args.threshold).float()
                    mask_g = max_pg.ge(args.threshold).float()

                tl_oh = F.one_hot(target_l, args.num_classes).float()
                tg_oh = F.one_hot(target_g, args.num_classes).float()
                # SAGE CDSC Equation 3: lambda = exp(-kappa * delta_C)
                # Paper default kappa=0.5; dusuk delta -> lambda yakın 1 (local'e güven)
                #                          yüksek delta -> lambda yakın 0 (global'e güven)
                delta = torch.clamp(torch.abs(max_pl - max_pg) + 1e-6, 1e-6, 1.0)
                lam   = torch.clamp(
                    torch.exp(-args.kappa * delta), 1e-6, 1.0)

                final_t = torch.where(
                    mask_l.unsqueeze(1).bool(),
                    lam.unsqueeze(1) * tl_oh + (1 - lam).unsqueeze(1) * tg_oh,
                    tg_oh,
                )
                mask_valid = torch.max(mask_l, mask_g)

                logits_s_prob = torch.softmax(logits_u_s, dim=-1) + 1e-10
                Lu = (F.kl_div(logits_s_prob.log(), final_t + 1e-10, reduction='none')
                      .sum(-1) * mask_valid).mean()

                loss = Lx + args.lambda_u * Lu
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), 10.0)
                self.optimizer.step()

                total_lx += Lx.item()
                total_lu += Lu.item()
                total_batches += 1
                epoch_pseudo.extend(target_l[mask_valid.bool()].cpu().numpy().tolist())

        final_state = {k: v.cpu() for k, v in self.local_model.state_dict().items()}
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        return (final_state, dict(Counter(epoch_pseudo)),
                total_lx / max(total_batches, 1),
                total_lu / max(total_batches, 1))

    def step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    @staticmethod
    def interleave(x, size):
        s = list(x.shape)
        return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

    @staticmethod
    def de_interleave(x, size):
        s = list(x.shape)
        return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


# ══════════════════════════════════════════════════════════════
# 8. YARDIMCI
# ══════════════════════════════════════════════════════════════

class _SubsetImageFolder(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices, transform=None):
        self.base_dataset  = base_dataset
        self.transform     = transform
        self.valid_indices = [i for i in indices if i < len(base_dataset)]
        self.targets       = [base_dataset.targets[i] for i in self.valid_indices]

    def __len__(self): return len(self.valid_indices)

    def __getitem__(self, idx):
        try:
            img, label = self.base_dataset[self.valid_indices[idx]]
            if self.transform: img = self.transform(img)
            return img, label
        except Exception:
            return self.__getitem__((idx + 1) % len(self.valid_indices))


# ══════════════════════════════════════════════════════════════
# 9. MAIN LOOP
# ══════════════════════════════════════════════════════════════

def main_loop(alpha):
    args = args_parser()
    args.alpha = alpha

    exp_name       = get_exp_name(args)
    local_ckpt_dir = os.path.join(args.checkpoint_dir, exp_name)

    os.makedirs(f'./results/{args.dataset}/logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(message)s',
        filename=f'./results/{args.dataset}/logs/{exp_name}.log', filemode='a',
    )

    # ── Veri Seti ────────────────────────────────────────────
    if args.dataset == 'HAM10000':
        args.num_classes = 7
        ham_mean, ham_std = [0.763, 0.545, 0.570], [0.140, 0.152, 0.169]
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=ham_mean, std=ham_std),
        ])
        full = ImageFolder(root=args.path_ham10000, transform=None)
        idx_tr, idx_te = train_test_split(range(len(full)), test_size=0.20,
                                          stratify=full.targets, random_state=args.seed)
        data_local_training = _SubsetImageFolder(full, idx_tr)
        data_global_test    = _SubsetImageFolder(
            ImageFolder(root=args.path_ham10000, transform=transform_test), idx_te)
        print(f"[HAM10000] Train: {len(idx_tr)} | Test: {len(idx_te)}")
        tc = Counter(full.targets[i] for i in idx_tr)
        print(f"  Train dist: { {full.classes[k]: v for k, v in sorted(tc.items())} }")

    elif args.dataset == 'CIFAR10':
        args.num_classes = 10
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])
        data_local_training = datasets.CIFAR10(
            args.path_cifar10, train=True, download=True, transform=None)
        data_global_test    = datasets.CIFAR10(
            args.path_cifar10, train=False, download=True, transform=transform_test)
        print(f"[CIFAR10] Train: {len(data_local_training)} | Test: {len(data_global_test)}")
    else:
        raise ValueError(f"Desteklenmeyen dataset: {args.dataset}")

    os.makedirs(local_ckpt_dir, exist_ok=True)

    # ── Aktif Flagleri Yazdir ─────────────────────────────────
    print(f"\n{'='*65}")
    print(f"[EXP] {exp_name}")
    print(f"{'='*65}")
    print(f"  Model            : {args.model}")
    print(f"  Aggregation      : {args.aggregation_method} (samples={args.shapley_samples})")
    print(f"  --- Mevcut ---")
    print(f"  Focal Loss       : {'ON' if args.use_focal_loss else 'off'}")
    print(f"  STFL             : {'ON' if args.use_stfl else 'off'} (beta={args.stfl_beta})")
    print(f"  Weighted Sampler : {'ON' if args.use_weighted_sampler else 'off'}")
    print(f"  Medical Augment  : {'ON' if args.use_medical_augment else 'off'}")
    print(f"  Cosine LR        : {'ON' if args.use_cosine_lr else 'off'}")
    print(f"  GroupNorm        : {'ON' if args.use_groupnorm else 'off'}")
    print(f"  --- Yeni ShapFed ---")
    print(f"  EMA Eval         : {'ON' if args.use_ema_eval else 'off'} (decay={args.ema_decay})")
    print(f"  Shapley EMA      : {'ON' if args.use_shapley_ema else 'off'} (decay={args.shapley_ema_decay})")
    print(f"  Personalized Init: {'ON' if args.use_personalized_init else 'off'} (strength={args.personalization_strength})")
    print(f"{'='*65}\n")

    # ── Veri Dagitimy ────────────────────────────────────────
    rng = np.random.RandomState(args.seed)
    list_label2indices = classify_label(data_local_training, args.num_classes)

    # IPC hesapla: num_labeled verilmisse sabit kullan,
    # yoksa label_ratio ile her sinif icin ayri hesapla
    if args.num_labeled is not None:
        # Sabit IPC modu (eski davranis)
        ipc = args.num_labeled
        ipc_per_class = [ipc] * args.num_classes
        print(f"[DATA] Sabit IPC={ipc} | Toplam etiketli~{ipc * args.num_classes}")
    else:
        # Oransal mod: her sinifin label_ratio kadar labeled
        ipc_per_class = []
        for c in range(args.num_classes):
            n_class = len(list_label2indices[c])
            n_labeled = max(1, int(n_class * args.label_ratio))
            # Guvence: en az 1, en fazla sinif buyuklugu kadar
            n_labeled = min(n_labeled, n_class)
            ipc_per_class.append(n_labeled)
        ipc = ipc_per_class  # partition_train listeli IPC alacak sekilde guncellendi
        names_tmp = CLASS_NAMES_HAM if args.num_classes == 7 else [str(c) for c in range(args.num_classes)]
        print(f"[DATA] Oransal IPC (ratio={args.label_ratio}):")
        total_labeled = 0
        for c, n in enumerate(ipc_per_class):
            n_total = len(list_label2indices[c])
            pct = n / n_total * 100
            cname = names_tmp[c] if c < len(names_tmp) else str(c)
            print(f"  {cname:<6}: {n:>4}/{n_total:<5} ({pct:.1f}% labeled)")
            total_labeled += n
        print(f"  Toplam labeled: {total_labeled}")

    # partition_train: sabit IPC veya per-class IPC listesi destekler
    if isinstance(ipc, list):
        l_lab, l_unlab = partition_train_perclass(list_label2indices, ipc)
    else:
        l_lab, l_unlab = partition_train(list_label2indices, ipc)
    if alpha == 0:
        c_lab   = clients_indices_homo(l_lab,   args.num_classes, args.num_clients)
        c_unlab = clients_indices_homo(l_unlab, args.num_classes, args.num_clients)
    else:
        c_lab   = clients_indices(l_lab,   args.num_classes, args.num_clients, alpha, seed=0)
        c_unlab = clients_indices(l_unlab, args.num_classes, args.num_clients, alpha, seed=0)

    for i in range(len(c_lab)):
        if i < len(c_unlab): c_unlab[i].extend(c_lab[i])
        else: c_unlab.append(list(c_lab[i]))

    # ── Modeller ─────────────────────────────────────────────
    global_model = Global(args)
    local_model  = Local(args)

    start_round, metrics_history = load_checkpoint(
        global_model.model, local_model.scheduler, local_ckpt_dir, args,
        global_model_ref=global_model)

    idx_labeled   = Indices2Dataset_labeled(
        data_local_training, dataset_name=args.dataset,
        use_medical_augment=args.use_medical_augment)
    idx_unlabeled = Indices2Dataset_unlabeled_fixmatch(
        data_local_training, dataset_name=args.dataset,
        use_medical_augment=args.use_medical_augment)

    dashboard_data = {}
    os.makedirs(f'./results/{args.dataset}', exist_ok=True)
    total_clients = list(range(args.num_clients))
    names = CLASS_NAMES_HAM if args.num_classes == 7 else [str(c) for c in range(args.num_classes)]

    print(f"[TRAIN] Round {start_round} -> {args.num_rounds}")

    for r in tqdm(range(start_round, args.num_rounds + 1), desc='Server'):
        dict_global_params = global_model.download_params()
        online = rng.choice(total_clients, args.num_online_clients, replace=False)

        list_params, list_nums = [], []
        round_dists, round_pseudo = {}, Counter()

        for client in online:
            idx_labeled.load(c_lab[client])
            idx_unlabeled.load(c_unlab[client])

            lbl_counts = Counter(data_local_training.targets[i]
                                 for i in c_lab[client] if i < len(data_local_training))
            round_dists[str(client)] = {str(k): v for k, v in lbl_counts.items()}

            summary = ', '.join(
                f"{names[int(k)] if int(k) < len(names) else k}: {v}"
                for k, v in sorted(lbl_counts.items()))
            print(f"\n> Client {client}  [LR={local_model.get_lr():.6f}]")
            print(f"    Labeled  : [{summary}]")
            print(f"    Unlabeled: {len(c_unlab[client])} adet")

            list_nums.append(len(c_lab[client]) + len(c_unlab[client]))

            # V3: Personalized init parametrelerini al
            init_params = local_model.get_init_params(
                client, copy.deepcopy(dict_global_params), args, global_model)

            params, pseudo, lx, lu = local_model.fixmatch_train(
                args, idx_labeled, idx_unlabeled, init_params, r)

            # V3: Client parametrelerini sakla (sonraki round icin)
            if args.use_personalized_init:
                local_model.store_client_params(client, params)

            list_params.append(copy.deepcopy(params))
            round_pseudo.update(pseudo)

            p_summary = ', '.join(
                f"{names[int(k)] if int(k) < len(names) else k}: {v}"
                for k, v in sorted(pseudo.items()))
            print(f"    Lx: {lx:.4f} | Lu: {lu:.4f}")
            print(f"    Pseudo: [{p_summary}]")
            del params
            torch.cuda.empty_cache()

        # Aggregation
        fedavg_params = global_model.initialize_for_model_fusion(
            args, list_params, list_nums, dict_global_params,
            online_client_ids=list(online))
        global_model.model.load_state_dict(fedavg_params)

        # V1: EMA guncelle
        global_model.update_ema(fedavg_params)

        # Eval: EMA aktifse ema_params kullan
        if args.use_ema_eval and global_model.ema_params is not None:
            eval_params = {k: v.to(fedavg_params[k].dtype)
                           for k, v in global_model.ema_params.items()}
        else:
            eval_params = fedavg_params

        acc, acsa, f1, pred_dist, per_class = global_model.fedavg_eval(
            copy.deepcopy(eval_params), data_global_test, args.batch_size_test, args)

        metrics_history['acc'].append(acc)
        metrics_history['acsa'].append(acsa)
        metrics_history['f1'].append(f1)

        b_acc  = max(metrics_history['acc'])
        b_acsa = max(metrics_history['acsa'])
        b_f1   = max(metrics_history['f1'])

        print(f"\n[Round {r:>4}/{args.num_rounds}] "
              f"Acc:{acc:.4f} ACSA:{acsa:.4f} F1:{f1:.4f} || "
              f"Best-> Acc:{b_acc:.4f} ACSA:{b_acsa:.4f} F1:{b_f1:.4f}")
        logging.info(f"Round {r} | Acc:{acc:.4f} ACSA:{acsa:.4f} F1:{f1:.4f} | "
                     f"Best ACSA:{b_acsa:.4f} | LR:{local_model.get_lr():.6f} | {per_class}")

        # Dashboard
        dashboard_data[str(r)] = {
            'client_distributions': round_dists,
            'pseudo_labels': {str(k): v for k, v in round_pseudo.items()},
            'global_predictions': {str(k): v for k, v in pred_dist.items()},
            'per_class_recall': per_class,
            'lr': local_model.get_lr(),
        }
        log_path = f'./results/{args.dataset}/dashboard_data.json'
        with open(log_path, 'w') as f:
            json.dump(dashboard_data, f, indent=2)
        try:
            boto3.client('s3').upload_file(
                log_path, args.s3_bucket, f"{exp_name}/dashboard_data.json")
        except Exception as e:
            print(f"[S3-WARN] {e}")

        # Checkpoint — yeni state'leri de kaydet
        extra = {
            'ema_params':           global_model.ema_params,
            'shapley_ema_store':    global_model.shapley_ema_store,
            'client_gamma': global_model.client_gamma,
        }
        save_checkpoint(
            r, global_model.download_params(),
            local_model.scheduler.state_dict() if local_model.scheduler else {},
            metrics_history, local_ckpt_dir, args, extra_state=extra)

        # CSV
        n = len(metrics_history['acc'])
        pd.DataFrame({
            'round': range(1, n + 1),
            'acc':   metrics_history['acc'],
            'acsa':  metrics_history['acsa'],
            'f1':    metrics_history['f1'],
        }).to_csv(f'./results/{args.dataset}/{exp_name}.csv', index=False)

        local_model.step_scheduler()


# ══════════════════════════════════════════════════════════════
# 10. ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    torch.backends.cudnn.deterministic = True
    args = args_parser()
    main_loop(args.alpha)
