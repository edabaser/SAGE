"""
SageMaker Training Job entry point.
- S3'teki HAM10000 dataseti /tmp/data/HAM10000'e sync'ler
- Sonra SAGE_ShapFed_HAM10000.py'nin main_loop'unu çağırır
- Checkpoint ve sonuçları S3'e yazar
"""

import os
import sys
import subprocess
import argparse
import shutil

# ── SageMaker environment path'lerini ayarla ──────────────────
SM_CHANNEL_DATASET = os.environ.get('SM_CHANNEL_DATASET', '/opt/ml/input/data/dataset')
SM_OUTPUT_DATA_DIR = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data')
SM_MODEL_DIR       = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')

# ── Argümanlar ─────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',            type=str,   default='HAM10000')
parser.add_argument('--aggregation_method', type=str,   default='ShapFed')
parser.add_argument('--alpha',              type=float, default=1.0)
parser.add_argument('--gpu_id',             type=int,   default=0)
parser.add_argument('--num_clients',        type=int,   default=20)
parser.add_argument('--num_online_clients', type=int,   default=8)
parser.add_argument('--local_epochs',       type=int,   default=5)
parser.add_argument('--shapley_samples',    type=int,   default=10)
parser.add_argument('--threshold',          type=float, default=0.95)
parser.add_argument('--lambda_u',           type=float, default=1.0)
parser.add_argument('--T',                  type=float, default=1.0)
parser.add_argument('--mu',                 type=int,   default=2)
parser.add_argument('--seed',               type=int,   default=7)
parser.add_argument('--batch_size_local_labeled_fixmatch', type=int, default=24)
parser.add_argument('--batch_size_local_labeled',          type=int, default=24)
parser.add_argument('--batch_size_local_unlabeled',        type=int, default=24)
parser.add_argument('--batch_size_test',                   type=int, default=128)
parser.add_argument('--lr_local_training',       type=float, default=0.03)
parser.add_argument('--lr_distillation_training',type=float, default=0.01)
parser.add_argument('--kappa',              type=float, default=0.5)
args, _ = parser.parse_known_args()

# ── Dataset path ───────────────────────────────────────────────
# SageMaker input channel olarak gelir, direkt kullan
DATA_DIR       = SM_CHANNEL_DATASET          # /opt/ml/input/data/dataset
CKPT_DIR       = '/tmp/checkpoints'
RESULTS_DIR    = '/tmp/results'

os.makedirs(CKPT_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"[INFO] Dataset dir  : {DATA_DIR}")
print(f"[INFO] Checkpoint   : {CKPT_DIR}")
print(f"[INFO] Results      : {RESULTS_DIR}")
print(f"[INFO] Dataset contents: {os.listdir(DATA_DIR)}")

# ── sys.argv'yi ana script için yeniden oluştur ───────────────
sys.argv = [
    'SAGE_ShapFed_HAM10000.py',
    '--dataset',            args.dataset,
    '--path_ham10000',      DATA_DIR,
    '--checkpoint_dir',     CKPT_DIR,
    '--aggregation_method', args.aggregation_method,
    '--alpha',              str(args.alpha),
    '--gpu_id',             str(args.gpu_id),
    '--num_clients',        str(args.num_clients),
    '--num_online_clients', str(args.num_online_clients),
    '--local_epochs',       str(args.local_epochs),
    '--shapley_samples',    str(args.shapley_samples),
    '--threshold',          str(args.threshold),
    '--lambda_u',           str(args.lambda_u),
    '--T',                  str(args.T),
    '--mu',                 str(args.mu),
    '--seed',               str(args.seed),
    '--batch_size_local_labeled_fixmatch', str(args.batch_size_local_labeled_fixmatch),
    '--batch_size_local_labeled',          str(args.batch_size_local_labeled),
    '--batch_size_local_unlabeled',        str(args.batch_size_local_unlabeled),
    '--batch_size_test',                   str(args.batch_size_test),
    '--lr_local_training',                 str(args.lr_local_training),
    '--lr_distillation_training',          str(args.lr_distillation_training),
    '--kappa',                             str(args.kappa),
]

# ── Ana training loop'u çalıştır ──────────────────────────────
from SAGE_ShapFed_HAM10000 import main_loop
main_loop(args.alpha)

# ── Çıktıları SageMaker output dizinine kopyala ───────────────
print("[INFO] Copying results to output dir...")
if os.path.exists(RESULTS_DIR):
    shutil.copytree(RESULTS_DIR, os.path.join(SM_OUTPUT_DATA_DIR, 'results'), dirs_exist_ok=True)

if os.path.exists(CKPT_DIR):
    shutil.copytree(CKPT_DIR, os.path.join(SM_MODEL_DIR, 'checkpoints'), dirs_exist_ok=True)

print("[INFO] Training complete.")
