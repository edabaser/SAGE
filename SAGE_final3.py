"""
SAGE_final2.py — Federated Semi-Supervised Learning with ShapFed Integration
=============================================================================
Combines:
  - SAGE (Liu et al., CVPR 2025): Confidence-Discrepancy pseudo-label correction
  - ShapFed (Tastan et al., IJCAI 2024): Class-Specific Shapley Value aggregation

Architecture:
  - Global model: server-side aggregation & evaluation
  - Local model: client-side FixMatch training with SAGE pseudo-label correction
  - ShapFed CSSV: gradient-based cosine similarity (O(n+1), no Monte Carlo)

Flags (all togglable via CLI args):
  --use_focal_loss        Focal Loss for class imbalance (required for HAM10000)
  --use_stfl              STFL dynamic threshold (rare classes get lower threshold)
  --use_medical_augment   Color-preserving augmentation for dermoscopy images
  --use_groupnorm         BatchNorm -> GroupNorm conversion (prevents stat poisoning)
  --use_cosine_lr         Cosine Annealing LR scheduler
  --use_ema_eval          EMA-smoothed model weights for evaluation
  --use_shapley_ema       EMA smoothing on CSSV across rounds (mu=0.9)
  --use_personalized_init Personalized broadcast: w_bar_i = gamma_i*w_s + (1-gamma_i)*w_i

Checkpoint naming:
  {dataset}_a{alpha}_{agg}_{model}_{flags}_L{ipc}_C{online}_E{epochs}_T{thr}_LR{lr}
  Flags: FL STFL MA CLR GN EMA SEMA PINIT
"""

import os
import copy
import random
import logging
import json
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import torchvision.models as tv_models
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # Handle corrupted JPEG files in HAM10000

# S3 support (optional — gracefully disabled if boto3 not installed)
try:
    import boto3
    from botocore.exceptions import ClientError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

from options import args_parser
from Dataset.dataset import (
    classify_label,
    Indices2Dataset_labeled,
    Indices2Dataset_unlabeled_fixmatch,
    partition_train,
)
from Dataset.sample_dirichlet import clients_indices, clients_indices_homo

# HAM10000 class names in sorted order (ImageFolder alphabetical)
CLASS_NAMES_HAM = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


# =============================================================================
# 1. DATASET UTILITIES
# =============================================================================

def partition_train_perclass(list_label2indices, ipc_per_class):
    """
    Variant of partition_train that accepts a per-class IPC list.
    ipc_per_class: list of length num_classes, ipc_per_class[c] = labeled count for class c.
    Returns (list_labeled_indices, list_unlabeled_indices) one entry per class.
    """
    list_label2indices_labeled = []
    list_label2indices_unlabeled = []
    for c, indices in enumerate(list_label2indices):
        ipc_c = int(ipc_per_class[c]) if c < len(ipc_per_class) else int(ipc_per_class[-1])
        idx_shuffle = np.random.permutation(indices)
        labeled = list(idx_shuffle[:ipc_c])
        unlabeled = list(idx_shuffle[ipc_c:])
        list_label2indices_labeled.append(labeled)
        list_label2indices_unlabeled.append(unlabeled)
    return list_label2indices_labeled, list_label2indices_unlabeled


class _SubsetImageFolder(torch.utils.data.Dataset):
    """
    A subset wrapper over ImageFolder that:
      - restricts to given indices
      - applies an optional transform at __getitem__ time
      - exposes .targets for downstream label counting
    """
    def __init__(self, base_dataset, indices, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform
        # Clamp indices to valid range (guards against off-by-one in splits)
        self.valid_indices = [i for i in indices if i < len(base_dataset)]
        self.targets = [base_dataset.targets[i] for i in self.valid_indices]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Gracefully skip corrupted images
        try:
            img, label = self.base_dataset[self.valid_indices[idx]]
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception:
            return self.__getitem__((idx + 1) % len(self.valid_indices))


# =============================================================================
# 2. MODEL FACTORY
# =============================================================================

def _convert_bn_to_gn(model, num_groups=16):
    """
    Recursively replace all BatchNorm2d with GroupNorm(num_groups, nc).
    Pre-trained affine parameters (weight, bias) are copied to preserve
    ImageNet features — critical when starting from torchvision pretrained weights.

    If nc is not divisible by num_groups, falls back to the largest divisor of nc
    in [8, 4, 2, 1] to avoid runtime errors.
    """
    for name, child in list(model.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            nc = child.num_features
            g = num_groups
            # Find a valid group count
            if nc % g != 0:
                for fg in [8, 4, 2, 1]:
                    if nc % fg == 0:
                        g = fg
                        break
            gn = nn.GroupNorm(g, nc, affine=True, eps=1e-5)
            # Copy pretrained affine parameters
            if child.weight is not None:
                with torch.no_grad():
                    gn.weight.copy_(child.weight)
            if child.bias is not None:
                with torch.no_grad():
                    gn.bias.copy_(child.bias)
            setattr(model, name, gn)
        else:
            _convert_bn_to_gn(child, num_groups)


def build_model(args):
    """
    Factory that returns the correct model for args.model.

    resnet8  : Lightweight model from original SAGE codebase (CIFAR-scale)
    resnet18 : Pretrained torchvision ResNet-18, fc replaced for HAM10000

    GroupNorm conversion is applied when --use_groupnorm is set.
    For resnet18 WITHOUT GroupNorm, BatchNorm layers are frozen in eval mode
    to avoid batch-statistics corruption under Non-IID federated training.
    """
    if args.model == 'resnet8':
        from Model.resnet import ResNet
        model = ResNet(
            resnet_size=8, scaling=4,
            save_activations=False,
            group_norm_num_groups=args.group_norm_num_groups if args.use_groupnorm else None,
            freeze_bn=False, freeze_bn_affine=False,
            num_classes=args.num_classes,
        )
        # HAM images are 224x224; resnet8 needs adaptive pooling to avoid size mismatch
        if args.dataset == 'HAM10000':
            model.avgpool = nn.AdaptiveAvgPool2d(1)

    elif args.model == 'resnet18':
        from torchvision.models import ResNet18_Weights
        model = tv_models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Replace final fc layer for target number of classes
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
        if args.use_groupnorm:
            # BN -> GN with pretrained weight copy (see function above)
            _convert_bn_to_gn(model, args.group_norm_num_groups)
        else:
            # Freeze BN stats to prevent Non-IID poisoning of batch statistics
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
    else:
        raise ValueError(f"Unknown model: {args.model}. Choose 'resnet8' or 'resnet18'.")

    return model


def forward_model(model, x):
    """
    Unified forward pass. Original SAGE resnet8 returns (features, logits).
    torchvision ResNet-18 returns only logits.
    Always returns (features_or_None, logits).
    """
    out = model(x)
    if isinstance(out, tuple):
        return out  # (features, logits)
    return None, out  # (None, logits)


def _get_classifier_keys(args):
    """Return (weight_key, bias_key) for the final classification layer."""
    if args.model == 'resnet18':
        return 'fc.weight', 'fc.bias'
    # resnet8 from SAGE uses 'classifier'
    return 'classifier.weight', 'classifier.bias'


# =============================================================================
# 3. LOSS FUNCTIONS
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) for class imbalance.
    FL(p_t) = -(1 - p_t)^gamma * alpha_t * log(p_t)

    alpha_weights: per-class inverse-frequency weights (computed in build_criterion)
    gamma: focusing parameter; gamma=2 is the default from the paper.
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
        return focal.mean() if self.reduction == 'mean' else focal.sum()


def build_criterion(class_counts, num_classes, device, use_focal):
    """
    Build loss criterion for a single client.
    class_counts: dict {class_id: count} from this client's labeled data.
    When use_focal=True, inverse-frequency weights are computed and normalized
    so the mean weight = 1.0 (avoids LR scaling issues).
    """
    if not use_focal:
        return nn.CrossEntropyLoss().to(device)
    total = sum(class_counts.values()) + 1e-8
    # Inverse frequency: classes with fewer samples get higher weight
    weights = [total / (num_classes * max(class_counts.get(c, 1), 1))
               for c in range(num_classes)]
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    w = w / w.mean()  # Normalize so mean=1 (preserves effective LR)
    return FocalLoss(alpha_weights=w, gamma=2.0)


# =============================================================================
# 4. AUGMENTATION
# =============================================================================

class MedicalAugment:
    """
    Color-preserving augmentation pipeline for dermoscopy images.
    Avoids color-altering operations (hue jitter, grayscale) that remove
    diagnostically relevant color information from skin lesion images.

    Weak augmentation: mild flip + crop (used for pseudo-label generation).
    Strong augmentation: aggressive spatial transforms (used for consistency loss).
    """

    @staticmethod
    def weak(img_size=224):
        return transforms.Compose([
            transforms.Resize((img_size + 16, img_size + 16)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.763, 0.545, 0.570], [0.140, 0.152, 0.169]),
        ])

    @staticmethod
    def strong(img_size=224):
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            # Brightness/contrast only — no hue change
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
            transforms.RandomAffine(degrees=0, shear=15),
            transforms.ToTensor(),
            transforms.Normalize([0.763, 0.545, 0.570], [0.140, 0.152, 0.169]),
        ])


def get_transforms(args):
    """
    Return (transform_labeled, transform_unlabeled_weak, transform_unlabeled_strong, transform_test)
    based on dataset and flags.
    """
    if args.dataset == 'HAM10000':
        if args.use_medical_augment:
            t_lab = MedicalAugment.weak()
            t_unlab_w = MedicalAugment.weak()
            t_unlab_s = MedicalAugment.strong()
        else:
            base = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.763, 0.545, 0.570], [0.140, 0.152, 0.169]),
            ])
            t_lab = t_unlab_w = t_unlab_s = base
        t_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.763, 0.545, 0.570], [0.140, 0.152, 0.169]),
        ])
    elif args.dataset == 'CIFAR10':
        # Standard FixMatch augmentations for CIFAR-10
        t_lab = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])
        t_unlab_w = t_lab
        t_unlab_s = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])
        t_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    return t_lab, t_unlab_w, t_unlab_s, t_test


# =============================================================================
# 5. EXPERIMENT NAMING & CHECKPOINTING
# =============================================================================

def get_exp_name(args):
    """
    Deterministic experiment name encoding all key hyperparameters.
    Used for checkpoint paths, CSV files, and S3 keys.
    """
    flags = []
    if args.use_focal_loss:      flags.append('FL')
    if args.use_stfl:            flags.append('STFL')
    if args.use_medical_augment: flags.append('MA')
    if args.use_cosine_lr:       flags.append('CLR')
    if args.use_groupnorm:       flags.append('GN')
    if args.use_ema_eval:        flags.append('EMA')
    if args.use_shapley_ema:     flags.append('SEMA')
    if args.use_personalized_init: flags.append('PINIT')
    flag_str = '_'.join(flags) if flags else 'BASE'
    return (
        f"{args.dataset}_a{args.alpha}_{args.aggregation_method}_{args.model}_"
        f"{flag_str}_L{args.num_labeled}_C{args.num_online_clients}_"
        f"E{args.local_epochs}_T{args.threshold}_LR{args.lr_local_training}"
    )


def save_checkpoint(round_num, model_state, scheduler_state, metrics_history,
                    local_ckpt_dir, args, filename='checkpoint.pt',
                    backup_every=5, extra_state=None):
    """
    Save training checkpoint locally, then optionally sync to S3.
    extra_state: dict of additional tensors/dicts (EMA params, Shapley store, etc.)
    """
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

    # S3 backup every backup_every rounds and at the final round
    if HAS_BOTO3 and (round_num % backup_every == 0 or round_num == args.num_rounds):
        folder_name = get_exp_name(args)
        try:
            s3 = boto3.client('s3')
            s3.upload_file(local_path, args.s3_bucket,
                           f"checkpoints/{folder_name}/{filename}")
            # Also upload the metrics CSV if it exists
            local_csv = f'./results/{args.dataset}/{get_exp_name(args)}.csv'
            if os.path.exists(local_csv):
                s3.upload_file(local_csv, args.s3_bucket,
                               f"results/{folder_name}/metrics.csv")
            print(f"[S3] Round {round_num} backed up.")
        except Exception as e:
            print(f"[S3-WARN] {e}")


def load_checkpoint(model, scheduler, local_ckpt_dir, args,
                    filename='checkpoint.pt', global_model_ref=None):
    """
    Load checkpoint from local disk; falls back to S3 download if not found locally.
    Returns (start_round, metrics_history).
    start_round = saved_round + 1, so training resumes from the next round.
    """
    folder_name = get_exp_name(args)
    local_path = os.path.join(local_ckpt_dir, filename)

    if not os.path.exists(local_path):
        if HAS_BOTO3:
            s3_path = f"checkpoints/{folder_name}/{filename}"
            try:
                s3 = boto3.client('s3')
                print(f"[CKPT] Downloading from S3: {s3_path}")
                os.makedirs(local_ckpt_dir, exist_ok=True)
                s3.download_file(args.s3_bucket, s3_path, local_path)
            except Exception:
                print("[CKPT] Not found. Starting from scratch.")
                return 1, {'acc': [], 'acsa': [], 'f1': []}
        else:
            print("[CKPT] No checkpoint found. Starting from scratch.")
            return 1, {'acc': [], 'acsa': [], 'f1': []}

    print(f"[CKPT] Loading: {local_path}")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ckpt = torch.load(local_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        # Restore optional EMA/Shapley state
        if global_model_ref is not None:
            if 'ema_params' in ckpt and ckpt['ema_params'] is not None:
                global_model_ref.ema_params = ckpt['ema_params']
                print("[CKPT] EMA params restored.")
            if 'shapley_ema_store' in ckpt and ckpt['shapley_ema_store']:
                global_model_ref.shapley_ema_store = ckpt['shapley_ema_store']
                print("[CKPT] Shapley EMA store restored.")
            if 'client_gamma' in ckpt and ckpt['client_gamma']:
                global_model_ref.client_gamma = ckpt['client_gamma']
                print("[CKPT] Client gamma scores restored.")
        return ckpt['round'] + 1, ckpt['metrics_history']
    except Exception as e:
        print(f"[CKPT] Load error: {e}. Starting from scratch.")
        return 1, {'acc': [], 'acsa': [], 'f1': []}


# =============================================================================
# 6. SHAPLEY (CSSV) — ShapFed Paper Equation 5
# =============================================================================

def compute_cssv(args, local_models_params, initial_global_params):
    """
    Compute Class-Specific Shapley Values (CSSVs) using gradient-based cosine
    similarity as in ShapFed paper Equation 5.

    Implementation: O(n+1) — NO Monte Carlo sampling.
    Steps:
      1. Compute per-client gradient = (global_params - local_params) / lr
         (equivalent to the pseudo-gradient that represents client's update direction)
      2. Compute aggregated gradient = weighted mean of all client gradients
         (this is the "coalition" gradient — the full coalition with all n clients)
      3. CSSV[i, j] = cosine_similarity(client_i_grad_class_j, agg_grad_class_j)
         This measures how aligned client i's update is with the consensus
         for class j.

    NOTE: We use (global - local) as gradient direction because the client
    moves AWAY from global_params during local SGD. The sign matters for
    cosine similarity: a client whose update aligns with the aggregate gets
    a positive similarity (positive contribution).

    Returns:
      cssv: np.array shape [num_clients, num_classes], values in [-1, 1]
      gamma: np.array shape [num_clients], per-client mean contribution in [0, 1]
             gamma_i = (1/M) * sum_j (1 + CSSV[i,j]) / 2
    """
    num_clients = len(local_models_params)
    num_classes = args.num_classes
    weight_key, bias_key = _get_classifier_keys(args)

    # ── Step 1: Compute per-client gradients (classifier layer only, CPU) ──
    g_w = initial_global_params[weight_key].float().cpu()  # [C, D]
    g_b = initial_global_params[bias_key].float().cpu()    # [C]

    client_grads = []  # list of dicts with weight_key, bias_key
    for lp in local_models_params:
        # gradient = global - local (direction of update = client's contribution)
        dw = g_w - lp[weight_key].float().cpu()  # [C, D]
        db = g_b - lp[bias_key].float().cpu()    # [C]
        client_grads.append({weight_key: dw, bias_key: db})

    # ── Step 2: Aggregate gradient (equal-weight mean = full-coalition gradient) ──
    agg_w = sum(cg[weight_key] for cg in client_grads) / num_clients  # [C, D]
    agg_b = sum(cg[bias_key] for cg in client_grads) / num_clients    # [C]

    # ── Step 3: Per-client, per-class cosine similarity ──
    cssv = np.zeros((num_clients, num_classes), dtype=np.float32)

    for i, cg in enumerate(client_grads):
        for c in range(num_classes):
            # Client vector for class c: [D+1]
            v_client = torch.cat([cg[weight_key][c].view(-1),
                                  cg[bias_key][c].view(-1)])
            # Aggregate vector for class c: [D+1]
            v_agg = torch.cat([agg_w[c].view(-1), agg_b[c].view(-1)])

            norm_c = torch.norm(v_client)
            norm_a = torch.norm(v_agg)

            if norm_c < 1e-8 or norm_a < 1e-8:
                # Zero-gradient client (no update for this class): neutral score
                cssv[i, c] = 0.0
            else:
                sim = F.cosine_similarity(
                    v_client.unsqueeze(0),
                    v_agg.unsqueeze(0)
                ).item()
                cssv[i, c] = sim  # in [-1, 1]

    # ── gamma_i: mean normalized CSSV across classes (Paper Eq.6) ──
    # Maps [-1, 1] -> [0, 1] via (1 + cssv) / 2
    gamma = np.mean((1.0 + cssv) / 2.0, axis=1)  # [num_clients]

    return cssv, gamma


def normalize_cssv_columns(cssv):
    """
    Column-normalize CSSV so each class column sums to 1.
    Negative values are clipped to 0 before normalization.
    This gives per-class aggregation weights: cssv_tilde[i, c] = cssv[i,c] / sum_i(cssv[i,c])
    (Paper Equation 7, class-specific part)
    """
    cssv_pos = np.maximum(cssv, 0.0)
    cssv_norm = cssv_pos.copy()
    num_classes = cssv.shape[1]
    num_clients = cssv.shape[0]
    for c in range(num_classes):
        col_sum = cssv_pos[:, c].sum()
        if col_sum > 1e-8:
            cssv_norm[:, c] = cssv_pos[:, c] / col_sum
        else:
            # Fallback: uniform weights if all clients have zero gradient for class c
            cssv_norm[:, c] = 1.0 / num_clients
    return cssv_norm


# =============================================================================
# 7. GLOBAL MODEL
# =============================================================================

class Global:
    """
    Server-side model. Handles:
      - Model aggregation (FedAvg or ShapFed)
      - EMA parameter tracking for evaluation (V1)
      - Shapley EMA smoothing across rounds (V2)
      - Per-client gamma tracking for personalized broadcast (V3)
    """

    def __init__(self, args):
        self.model = build_model(args)
        self.model.cuda(args.gpu_id)
        self.num_classes = args.num_classes
        self.args = args

        # V1: EMA of global model parameters for smoother evaluation
        self.ema_params = None
        self.ema_decay = getattr(args, 'ema_decay', 0.95)

        # V2: Per-client CSSV EMA store — smooths Shapley values across rounds
        # Key: str(client_id), Value: np.array [num_classes]
        self.shapley_ema_store = {}
        self.shapley_ema_decay = getattr(args, 'shapley_ema_decay', 0.9)

        # V3: Per-client gamma for personalized init (Paper Algorithm 1)
        # gamma_i ∈ [0,1]: how much global model to mix in for this client
        self.client_gamma = {}   # Not normalized — for personalized broadcast
        self.client_gamma_tilde = {}  # Normalized — for aggregation

    def update_ema(self, new_params):
        """
        EMA update: ema = decay * ema_prev + (1-decay) * new_params
        Parameters stored on CPU to avoid GPU memory fragmentation.
        """
        if self.ema_params is None:
            self.ema_params = {k: v.float().cpu().clone() for k, v in new_params.items()}
        else:
            for k in self.ema_params:
                self.ema_params[k] = (
                    self.ema_decay * self.ema_params[k].cpu() +
                    (1.0 - self.ema_decay) * new_params[k].float().cpu()
                )

    def initialize_for_model_fusion(self, args, list_dicts_local_params,
                                    list_nums_local_data, initial_global_params,
                                    online_client_ids=None):
        """
        Aggregate local model parameters into a new global model.

        FedAvg: weighted mean by number of local data samples.
        ShapFed: 
          - fc layer: class-specific CSSV-weighted aggregation (Paper Eq.7)
          - backbone: gamma_tilde-weighted aggregation (Paper Eq.7)

        Key correction vs. naive implementations:
          The fc layer uses column-normalized CSSV (one weight per class per client).
          The backbone uses gamma_tilde (scalar per client = mean CSSV contribution).
        """
        nc = len(list_dicts_local_params)
        total_data = sum(list_nums_local_data)
        weight_key, bias_key = _get_classifier_keys(args)

        if args.aggregation_method == 'ShapFed':
            # ── Compute CSSV ──────────────────────────────────────────────────
            cssv_raw, gamma_raw = compute_cssv(
                args, list_dicts_local_params, initial_global_params)

            # ── V2: CSSV EMA smoothing ────────────────────────────────────────
            if args.use_shapley_ema and online_client_ids is not None:
                cssv_smoothed = np.zeros_like(cssv_raw)
                for i, cid in enumerate(online_client_ids):
                    key = str(cid)
                    if key in self.shapley_ema_store:
                        cssv_smoothed[i] = (
                            self.shapley_ema_decay * self.shapley_ema_store[key] +
                            (1.0 - self.shapley_ema_decay) * cssv_raw[i]
                        )
                    else:
                        cssv_smoothed[i] = cssv_raw[i]
                    self.shapley_ema_store[key] = cssv_smoothed[i].copy()
                # Recompute gamma from smoothed CSSV
                gamma = np.mean((1.0 + cssv_smoothed) / 2.0, axis=1)
                cssv = cssv_smoothed
            else:
                cssv = cssv_raw
                gamma = gamma_raw

            # ── gamma_tilde: normalize for backbone aggregation (Paper Eq.6) ──
            gamma_sum = gamma.sum()
            if gamma_sum > 1e-8:
                gamma_tilde = gamma / gamma_sum
            else:
                gamma_tilde = np.ones(nc) / nc

            # ── V3: Store per-client gamma for personalized broadcast ─────────
            if args.use_personalized_init and online_client_ids is not None:
                mu = self.shapley_ema_decay if args.use_shapley_ema else 0.9
                for i, cid in enumerate(online_client_ids):
                    key = str(cid)
                    if key in self.client_gamma:
                        self.client_gamma[key] = (
                            mu * self.client_gamma[key] + (1.0 - mu) * float(gamma[i]))
                    else:
                        self.client_gamma[key] = float(gamma[i])

            # ── Column-normalize CSSV for fc aggregation ──────────────────────
            cssv_norm = normalize_cssv_columns(cssv)  # [nc, num_classes]

            # ── Aggregate parameters ──────────────────────────────────────────
            fused = copy.deepcopy(list_dicts_local_params[0])
            for name in list_dicts_local_params[0]:
                orig_dtype = list_dicts_local_params[0][name].dtype

                if name in (weight_key, bias_key):
                    # fc layer: class-specific weighted sum (Paper Eq.7 first term)
                    ft = torch.zeros_like(
                        list_dicts_local_params[0][name], dtype=torch.float32)
                    for c in range(args.num_classes):
                        for i in range(nc):
                            ft[c] += (list_dicts_local_params[i][name][c].float()
                                      * float(cssv_norm[i, c]))
                else:
                    # Backbone: gamma_tilde weighted (Paper Eq.7 second term)
                    ft = sum(
                        list_dicts_local_params[i][name].float() * float(gamma_tilde[i])
                        for i in range(nc)
                    )
                fused[name] = ft.to(orig_dtype)

        else:
            # ── Standard FedAvg ───────────────────────────────────────────────
            fused = copy.deepcopy(list_dicts_local_params[0])
            for name in list_dicts_local_params[0]:
                orig_dtype = list_dicts_local_params[0][name].dtype
                ft = sum(
                    list_dicts_local_params[i][name].float()
                    * (list_nums_local_data[i] / total_data)
                    for i in range(nc)
                )
                fused[name] = ft.to(orig_dtype)

        return fused

    def fedavg_eval(self, params, data_test, batch_size_test, args):
        """
        Evaluate global model on the test set.
        Reports: accuracy, ACSA (= macro recall = per-class mean recall), macro F1,
        prediction distribution, and per-class recall breakdown.
        """
        self.model.load_state_dict(params)
        self.model.eval()
        all_labels, all_preds = [], []

        with torch.no_grad():
            for imgs, labels in DataLoader(
                    data_test, batch_size_test, num_workers=0, pin_memory=False):
                imgs = imgs.cuda(args.gpu_id)
                _, logits = forward_model(self.model, imgs)
                _, preds = torch.max(logits, -1)
                all_labels.extend(labels.numpy())
                all_preds.extend(preds.cpu().numpy())

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        acc = (all_labels == all_preds).mean()
        # ACSA = Average Class-Specific Accuracy = macro-averaged recall
        acsa = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        pred_dist = dict(Counter(all_preds.tolist()))

        # Per-class recall from confusion matrix
        cm = confusion_matrix(all_labels, all_preds,
                              labels=list(range(self.num_classes)))
        names = CLASS_NAMES_HAM if self.num_classes == 7 else [str(c) for c in range(self.num_classes)]
        per_class = {}
        print("\n +-- Per-Class Recall --------------------------------")
        for c in range(self.num_classes):
            row_sum = cm[c].sum()
            rec = cm[c, c] / row_sum if row_sum > 0 else 0.0
            cname = names[c] if c < len(names) else str(c)
            per_class[cname] = round(rec, 4)
            bar = '#' * int(rec * 25)
            print(f" | {cname:<6}: {rec:.4f} {bar}")
        print(" +----------------------------------------------------\n")

        torch.cuda.empty_cache()
        return acc, acsa, macro_f1, pred_dist, per_class

    def download_params(self):
        """Return a CPU copy of current global model state dict."""
        return {k: v.cpu() for k, v in self.model.state_dict().items()}


# =============================================================================
# 8. LOCAL MODEL (CLIENT)
# =============================================================================

class Local:
    """
    Client-side model. Implements:
      - FixMatch semi-supervised training
      - SAGE confidence-discrepancy pseudo-label correction (CDSC)
      - STFL dynamic threshold for minority classes
      - Gradient clipping for training stability
      - Personalized init broadcast (V3)
    """

    def __init__(self, args):
        self.local_model = build_model(args)  # Trainable model
        self.local_G = build_model(args)       # Frozen global reference model
        self.local_model.cuda(args.gpu_id)
        self.local_G.cuda(args.gpu_id)

        self.optimizer = torch.optim.SGD(
            self.local_model.parameters(),
            lr=args.lr_local_training,
            momentum=0.9,
            weight_decay=1e-4,
        )
        if args.use_cosine_lr:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=args.num_rounds,
                eta_min=args.lr_local_training * getattr(args, 'lr_min_ratio', 0.01),
            )
        else:
            self.scheduler = None

        # V3: Previous round parameters per client for personalized init
        # Key: str(client_id), Value: state_dict on CPU
        self.client_prev_params = {}

    def get_init_params(self, client_id, global_params, args, global_model_ref):
        """
        Paper Algorithm 1, Line 6: Personalized broadcast
        w_bar_i = gamma_i * w_server + (1 - gamma_i) * w_prev_i

        gamma_i high → trust the global model more (client contributed well)
        gamma_i low  → preserve own parameters (client contributed little)

        Only active when --use_personalized_init AND aggregation=ShapFed.
        Falls back to pure global params for first round (no previous params).
        """
        if not args.use_personalized_init or args.aggregation_method != 'ShapFed':
            return copy.deepcopy(global_params)

        key = str(client_id)
        n = args.num_clients
        # Get stored gamma; default 1/n (Algorithm 1 Line 1 initialization)
        gamma = global_model_ref.client_gamma.get(key, 1.0 / n)
        gamma = float(np.clip(gamma * getattr(args, 'personalization_strength', 1.0),
                               0.0, 1.0))

        if key not in self.client_prev_params:
            # No previous params yet — send pure global
            return copy.deepcopy(global_params)

        prev = self.client_prev_params[key]
        personalized = {}
        for k in global_params:
            g_cpu = global_params[k].float().cpu()
            p_cpu = prev[k].float().cpu()
            personalized[k] = (gamma * g_cpu + (1.0 - gamma) * p_cpu).to(
                global_params[k].dtype)
        return personalized

    def store_client_params(self, client_id, local_params):
        """Save post-training client params for next round's personalized init."""
        self.client_prev_params[str(client_id)] = {
            k: v.cpu().clone() for k, v in local_params.items()
        }

    def fixmatch_train(self, args, data_labeled, data_unlabeled, init_params, r):
        """
        Local FixMatch training with SAGE pseudo-label correction.

        Pipeline per batch:
          1. Supervised loss Lx on labeled data (Focal or CE)
          2. Global model generates pseudo-labels (weak aug)
          3. Local model generates pseudo-labels (weak aug)
          4. SAGE CDSC (Eq.3): blend local/global based on confidence discrepancy
             lambda = exp(-kappa * |conf_local - conf_global|)
             final_target = lambda * one_hot(local) + (1-lambda) * one_hot(global)
          5. Consistency loss Lu: KL(strong_aug || final_target)
          6. Total loss: Lx + lambda_u * Lu
          7. Gradient clip + SGD step

        STFL (when active):
          Dynamic threshold per class. Rare classes (low EMA prob) get a
          LOWER threshold so more pseudo-labels are generated for them.
          CORRECT direction: dyn_thresh = threshold * (1 - beta * (1 - ema_norm))
          ema_norm[j] = class_ema_prob[j] / max(class_ema_prob)
          Rare class: ema_norm close to 0 → dyn_thresh close to threshold*(1-beta)

        Returns:
          (state_dict_cpu, pseudo_label_counts, avg_lx, avg_lu)
        """
        device = f'cuda:{args.gpu_id}'

        # Build class-specific criterion from this client's labeled data
        local_labels = [int(data_labeled.dataset.targets[
                               data_labeled.client_dataset[i][0]])
                        for i in range(len(data_labeled))]
        class_counts = dict(Counter(local_labels))
        criterion = build_criterion(
            class_counts, args.num_classes, device,
            use_focal=args.use_focal_loss)

        labeled_loader = DataLoader(
            data_labeled,
            sampler=RandomSampler(data_labeled),
            batch_size=args.batch_size_local_labeled_fixmatch,
            drop_last=True, num_workers=0, pin_memory=False,
        )
        unlabeled_loader = DataLoader(
            data_unlabeled,
            sampler=RandomSampler(data_unlabeled),
            batch_size=args.batch_size_local_labeled_fixmatch * args.mu,
            drop_last=True, num_workers=0, pin_memory=False,
        )

        # Load global params to both local model (trainable) and local_G (frozen reference)
        init_gpu = {k: v.to(device) for k, v in init_params.items()}
        self.local_model.load_state_dict(init_gpu)
        self.local_model.train()
        self.local_G.load_state_dict(init_gpu)
        self.local_G.eval()
        del init_gpu

        # EMA of per-class pseudo-label probabilities (for STFL dynamic threshold)
        # Shape: [num_classes], initialized to uniform
        class_probs_ema = torch.ones(args.num_classes, device=device) / args.num_classes

        # Limit iterations to avoid excessive compute for large unlabeled sets
        MAX_ITER = 30
        local_iter = max(
            min(int(len(data_unlabeled) / args.batch_size_local_labeled_fixmatch), MAX_ITER),
            5
        )

        total_lx, total_lu, total_batches = 0.0, 0.0, 0
        epoch_pseudo = []  # All pseudo-labels accepted this round (for logging)

        for _ in range(args.local_epochs):
            lab_iter = iter(labeled_loader)
            unlab_iter = iter(unlabeled_loader)

            for _ in range(local_iter):
                # ── Get labeled batch ──────────────────────────────────────
                try:
                    inputs_x, targets_x = next(lab_iter)
                except StopIteration:
                    lab_iter = iter(labeled_loader)
                    inputs_x, targets_x = next(lab_iter)

                # ── Get unlabeled batch ────────────────────────────────────
                try:
                    inputs_u_w, inputs_u_s, _ = next(unlab_iter)
                except StopIteration:
                    unlab_iter = iter(unlabeled_loader)
                    inputs_u_w, inputs_u_s, _ = next(unlab_iter)

                inputs_x = inputs_x.to(device)
                inputs_u_w = inputs_u_w.to(device)
                inputs_u_s = inputs_u_s.to(device)
                targets_x = targets_x.to(device)
                bs = inputs_x.shape[0]

                # ── Interleaved forward (saves memory vs. separate passes) ──
                combined = self.interleave(
                    torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1
                ).to(device)
                _, logits = forward_model(self.local_model, combined)
                logits = self.de_interleave(logits, 2 * args.mu + 1)
                logits_x = logits[:bs]
                logits_u_w_loc, logits_u_s = logits[bs:].chunk(2)
                del logits

                # ── Supervised loss ────────────────────────────────────────
                Lx = criterion(logits_x, targets_x)

                # ── Pseudo-label generation ────────────────────────────────
                with torch.no_grad():
                    # Global model pseudo-labels (frozen)
                    _, logits_g = forward_model(self.local_G, inputs_u_w)
                    pseudo_g = torch.softmax(logits_g / args.T, dim=-1)
                    max_pg, target_g = torch.max(pseudo_g, dim=-1)

                    # Local model pseudo-labels (detached)
                    pseudo_l = torch.softmax(logits_u_w_loc.detach() / args.T, dim=-1)
                    max_pl, target_l = torch.max(pseudo_l, dim=-1)

                # ── STFL: dynamic threshold per class ─────────────────────
                if args.use_stfl:
                    # Update EMA of class probability distribution
                    class_probs_ema = (class_probs_ema * 0.99 +
                                       pseudo_l.mean(0).detach() * 0.01)
                    # Normalize: max class has ema_norm=1, rare class has ema_norm≈0
                    ema_norm = class_probs_ema / (class_probs_ema.max() + 1e-8)
                    beta = args.stfl_beta
                    # CORRECT direction: rare class (low ema_norm) → lower threshold
                    # → more pseudo-labels generated for minority classes
                    dyn_thresh = args.threshold * (1.0 - beta * (1.0 - ema_norm))
                    dyn_thresh = torch.clamp(dyn_thresh, min=0.5, max=0.99)
                    mask_l = max_pl.ge(dyn_thresh[target_l]).float()
                    mask_g = max_pg.ge(dyn_thresh[target_g]).float()
                else:
                    mask_l = max_pl.ge(args.threshold).float()
                    mask_g = max_pg.ge(args.threshold).float()

                # ── SAGE CDSC: confidence-discrepancy soft correction ──────
                # Paper Eq.3: lambda = exp(-kappa * delta_C)
                # delta_C = |conf_local - conf_global|
                # High delta → models disagree → blend more toward global
                # Low delta  → models agree    → trust local pseudo-label
                delta_c = torch.clamp(
                    torch.abs(max_pl - max_pg) + 1e-6, min=1e-6, max=1.0)
                lam = torch.clamp(
                    torch.exp(-args.kappa * delta_c), min=1e-6, max=1.0)

                tl_oh = F.one_hot(target_l, args.num_classes).float()
                tg_oh = F.one_hot(target_g, args.num_classes).float()

                # Blend: when local mask is active, mix local and global
                # When local mask is inactive (low confidence), use global target
                final_t = torch.where(
                    mask_l.unsqueeze(1).bool(),
                    lam.unsqueeze(1) * tl_oh + (1.0 - lam).unsqueeze(1) * tg_oh,
                    tg_oh,
                )

                # Accept sample if EITHER local or global mask is active
                mask_valid = torch.max(mask_l, mask_g)

                # ── Consistency loss Lu (KL divergence) ───────────────────
                # KL(strong_aug_probs || blended_soft_target)
                logits_s_prob = torch.softmax(logits_u_s, dim=-1) + 1e-10
                final_t_safe = final_t + 1e-10  # Numerical stability
                Lu = (F.kl_div(logits_s_prob.log(), final_t_safe, reduction='none')
                      .sum(-1) * mask_valid).mean()

                loss = Lx + args.lambda_u * Lu

                # ── Optimizer step ─────────────────────────────────────────
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping prevents exploding gradients in early rounds
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), 10.0)
                self.optimizer.step()

                total_lx += Lx.item()
                total_lu += Lu.item()
                total_batches += 1

                # Log which classes got pseudo-labels this batch
                accepted_mask = mask_valid.bool()
                epoch_pseudo.extend(target_l[accepted_mask].cpu().numpy().tolist())

        # Move final params to CPU before returning (keeps GPU memory clean)
        final_state = {k: v.cpu() for k, v in self.local_model.state_dict().items()}
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        avg_lx = total_lx / max(total_batches, 1)
        avg_lu = total_lu / max(total_batches, 1)
        return final_state, dict(Counter(epoch_pseudo)), avg_lx, avg_lu

    def step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    @staticmethod
    def interleave(x, size):
        """
        Interleave labeled and unlabeled batches.
        Ensures BatchNorm (if used) sees a mix of both distributions.
        Input shape: [N] → output shape: [N] (same data, different order)
        """
        s = list(x.shape)
        return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

    @staticmethod
    def de_interleave(x, size):
        """Reverse of interleave — restores original ordering."""
        s = list(x.shape)
        return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


# =============================================================================
# 9. MAIN TRAINING LOOP
# =============================================================================

def main_loop(alpha):
    """
    Full federated semi-supervised training loop.

    Flow per round:
      1. Server downloads global params
      2. Select online_clients (random subset, no replacement)
      3. For each online client:
         a. Optionally compute personalized init params
         b. Run FixMatch training (SAGE pseudo-label correction)
         c. Collect local params and data counts
      4. Aggregate (FedAvg or ShapFed)
      5. Update global model + EMA
      6. Evaluate on test set (EMA params if flag set)
      7. Log metrics, save checkpoint, save CSV
    """
    args = args_parser()
    args.alpha = alpha
    exp_name = get_exp_name(args)
    local_ckpt_dir = os.path.join(args.checkpoint_dir, exp_name)

    os.makedirs(f'./results/{args.dataset}/logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        filename=f'./results/{args.dataset}/logs/{exp_name}.log',
        filemode='a',
    )

    # ── Dataset Setup ─────────────────────────────────────────────────────────
    _, t_unlab_w, t_unlab_s, t_test = get_transforms(args)
    t_lab, _, _, _ = get_transforms(args)  # labeled uses weak augmentation

    if args.dataset == 'HAM10000':
        args.num_classes = 7
        full = ImageFolder(root=args.path_ham10000, transform=None)
        # 80/20 stratified split — reproducible with args.seed
        idx_tr, idx_te = train_test_split(
            range(len(full)), test_size=0.20,
            stratify=full.targets, random_state=args.seed)
        data_local_training = _SubsetImageFolder(full, idx_tr, transform=None)
        data_global_test = _SubsetImageFolder(
            ImageFolder(root=args.path_ham10000, transform=t_test), idx_te)
        print(f"[HAM10000] Train: {len(idx_tr)} | Test: {len(idx_te)}")
        tc = Counter(full.targets[i] for i in idx_tr)
        print(f"  Train dist: { {full.classes[k]: v for k, v in sorted(tc.items())} }")

    elif args.dataset == 'CIFAR10':
        args.num_classes = 10
        data_local_training = datasets.CIFAR10(
            args.path_cifar10, train=True, download=True, transform=None)
        data_global_test = datasets.CIFAR10(
            args.path_cifar10, train=False, download=True, transform=t_test)
        print(f"[CIFAR10] Train: {len(data_local_training)} | Test: {len(data_global_test)}")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # ── Print Configuration ────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"[EXP] {exp_name}")
    print(f"{'=' * 65}")
    print(f"  Model          : {args.model}")
    print(f"  Aggregation    : {args.aggregation_method}")
    print(f"  Alpha (Non-IID): {alpha}")
    print(f"  IPC            : {args.num_labeled}")
    print(f"  Clients        : {args.num_clients} total | {args.num_online_clients} online/round")
    print(f"  ── Flags ──")
    print(f"  Focal Loss     : {'ON' if args.use_focal_loss else 'off'}")
    print(f"  STFL           : {'ON' if args.use_stfl else 'off'} (beta={args.stfl_beta})")
    print(f"  Medical Augment: {'ON' if args.use_medical_augment else 'off'}")
    print(f"  GroupNorm      : {'ON' if args.use_groupnorm else 'off'} (G={args.group_norm_num_groups})")
    print(f"  Cosine LR      : {'ON' if args.use_cosine_lr else 'off'}")
    print(f"  EMA Eval       : {'ON' if args.use_ema_eval else 'off'} (decay={args.ema_decay})")
    print(f"  Shapley EMA    : {'ON' if args.use_shapley_ema else 'off'} (mu={args.shapley_ema_decay})")
    print(f"  Personalized   : {'ON' if args.use_personalized_init else 'off'}")
    print(f"  kappa          : {args.kappa}")
    print(f"  threshold      : {args.threshold}")
    print(f"{'=' * 65}\n")

    # ── Data Partitioning ──────────────────────────────────────────────────────
    rng = np.random.RandomState(args.seed)
    list_label2indices = classify_label(data_local_training, args.num_classes)

    # IPC mode: fixed count per class
    ipc = args.num_labeled
    l_lab, l_unlab = partition_train(list_label2indices, ipc)
    print(f"[DATA] IPC={ipc} | Total labeled≈{ipc * args.num_classes}")

    # Distribute to clients
    if alpha == 0:
        c_lab = clients_indices_homo(l_lab, args.num_classes, args.num_clients)
        c_unlab = clients_indices_homo(l_unlab, args.num_classes, args.num_clients)
    else:
        c_lab = clients_indices(l_lab, args.num_classes, args.num_clients,
                                 alpha, seed=0)
        c_unlab = clients_indices(l_unlab, args.num_classes, args.num_clients,
                                   alpha, seed=0)

    # Add labeled indices into each client's unlabeled pool (labels stripped)
    # This is standard FSSL practice: labeled data is used for supervised loss only
    for i in range(len(c_lab)):
        if i < len(c_unlab):
            c_unlab[i].extend(c_lab[i])
        else:
            c_unlab.append(list(c_lab[i]))

    # ── Model Initialization ──────────────────────────────────────────────────
    global_model = Global(args)
    local_model = Local(args)

    start_round, metrics_history = load_checkpoint(
        global_model.model,
        local_model.scheduler,
        local_ckpt_dir, args,
        global_model_ref=global_model,
    )

    # Dataset wrappers for dynamic index loading
    idx_labeled = Indices2Dataset_labeled(
        data_local_training,
        dataset_name=args.dataset,
        use_medical_augment=args.use_medical_augment,
    )
    idx_unlabeled = Indices2Dataset_unlabeled_fixmatch(
        data_local_training,
        dataset_name=args.dataset,
        use_medical_augment=args.use_medical_augment,
    )

    total_clients = list(range(args.num_clients))
    names = CLASS_NAMES_HAM if args.num_classes == 7 else [str(c) for c in range(args.num_classes)]
    dashboard_data = {}
    os.makedirs(f'./results/{args.dataset}', exist_ok=True)

    print(f"[TRAIN] Starting from round {start_round} → {args.num_rounds}")

    # ── Main Round Loop ────────────────────────────────────────────────────────
    for r in tqdm(range(start_round, args.num_rounds + 1), desc='Server'):

        # Download current global model params (CPU, deep copy)
        dict_global_params = global_model.download_params()

        # Select online clients (random without replacement)
        online = rng.choice(total_clients, args.num_online_clients, replace=False)

        list_params, list_nums = [], []
        round_dists, round_pseudo = {}, Counter()

        # ── Client Training ────────────────────────────────────────────────
        for client in online:
            idx_labeled.load(c_lab[client])
            idx_unlabeled.load(c_unlab[client])

            lbl_counts = Counter(
                data_local_training.targets[i]
                for i in c_lab[client] if i < len(data_local_training))
            round_dists[str(client)] = {str(k): v for k, v in lbl_counts.items()}

            summary = ', '.join(
                f"{names[int(k)] if int(k) < len(names) else k}: {v}"
                for k, v in sorted(lbl_counts.items()))
            print(f"\n> Client {client} [LR={local_model.get_lr():.6f}]")
            print(f"  Labeled: [{summary}]")
            print(f"  Unlabeled: {len(c_unlab[client])} samples")

            list_nums.append(len(c_lab[client]) + len(c_unlab[client]))

            # V3: Personalized init for this client
            init_params = local_model.get_init_params(
                client, copy.deepcopy(dict_global_params), args, global_model)

            # Train
            params, pseudo, lx, lu = local_model.fixmatch_train(
                args, idx_labeled, idx_unlabeled, init_params, r)

            # V3: Store params for next round personalized init
            if args.use_personalized_init:
                local_model.store_client_params(client, params)

            list_params.append(copy.deepcopy(params))
            round_pseudo.update(pseudo)

            p_summary = ', '.join(
                f"{names[int(k)] if int(k) < len(names) else k}: {v}"
                for k, v in sorted(pseudo.items()))
            print(f"  Lx: {lx:.4f} | Lu: {lu:.4f}")
            print(f"  Pseudo accepted: [{p_summary}]")

            # Check for nv dominance in pseudo-labels (HAM10000 specific warning)
            if args.num_classes == 7 and pseudo:
                total_pl = sum(pseudo.values())
                nv_idx = CLASS_NAMES_HAM.index('nv')
                nv_count = pseudo.get(nv_idx, 0)
                if total_pl > 0 and nv_count / total_pl > 0.85:
                    print(f"  [WARN] nv dominance: {nv_count}/{total_pl} "
                          f"({100*nv_count/total_pl:.1f}%) — consider enabling STFL")

            del params
            torch.cuda.empty_cache()

        # ── Aggregation ────────────────────────────────────────────────────
        fedavg_params = global_model.initialize_for_model_fusion(
            args, list_params, list_nums,
            dict_global_params,
            online_client_ids=list(online),
        )
        global_model.model.load_state_dict(
            {k: v.to(f'cuda:{args.gpu_id}') for k, v in fedavg_params.items()})

        # V1: Update EMA of global model
        global_model.update_ema(fedavg_params)

        # ── Evaluation ────────────────────────────────────────────────────
        if args.use_ema_eval and global_model.ema_params is not None:
            eval_params = {
                k: v.to(fedavg_params[k].dtype).to(f'cuda:{args.gpu_id}')
                for k, v in global_model.ema_params.items()
            }
        else:
            eval_params = {
                k: v.to(f'cuda:{args.gpu_id}') for k, v in fedavg_params.items()
            }

        acc, acsa, f1, pred_dist, per_class = global_model.fedavg_eval(
            eval_params, data_global_test, args.batch_size_test, args)

        metrics_history['acc'].append(acc)
        metrics_history['acsa'].append(acsa)
        metrics_history['f1'].append(f1)

        b_acc = max(metrics_history['acc'])
        b_acsa = max(metrics_history['acsa'])
        b_f1 = max(metrics_history['f1'])

        print(f"\n[Round {r:>4}/{args.num_rounds}] "
              f"Acc:{acc:.4f} ACSA:{acsa:.4f} F1:{f1:.4f} || "
              f"Best→ Acc:{b_acc:.4f} ACSA:{b_acsa:.4f} F1:{b_f1:.4f}")
        logging.info(
            f"Round {r} | Acc:{acc:.4f} ACSA:{acsa:.4f} F1:{f1:.4f} | "
            f"Best ACSA:{b_acsa:.4f} | LR:{local_model.get_lr():.6f} | {per_class}")

        # ── Dashboard JSON ────────────────────────────────────────────────
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

        # ── Checkpoint ────────────────────────────────────────────────────
        extra = {
            'ema_params': global_model.ema_params,
            'shapley_ema_store': global_model.shapley_ema_store,
            'client_gamma': global_model.client_gamma,
        }
        save_checkpoint(
            r,
            {k: v.cpu() for k, v in global_model.model.state_dict().items()},
            local_model.scheduler.state_dict() if local_model.scheduler else {},
            metrics_history,
            local_ckpt_dir, args,
            extra_state=extra,
        )

        # ── CSV Metrics ───────────────────────────────────────────────────
        n = len(metrics_history['acc'])
        pd.DataFrame({
            'round': range(1, n + 1),
            'acc': metrics_history['acc'],
            'acsa': metrics_history['acsa'],
            'f1': metrics_history['f1'],
        }).to_csv(f'./results/{args.dataset}/{exp_name}.csv', index=False)

        # LR scheduler step (Cosine Annealing)
        local_model.step_scheduler()


# =============================================================================
# 10. ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    # Global seeds for full reproducibility
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = args_parser()
    main_loop(args.alpha)
