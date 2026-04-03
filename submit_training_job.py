"""
SageMaker Training Job submit scripti.
Bu dosyayı SageMaker notebook'unda bir hücrede çalıştır.

Kullanım:
    import submit_training_job
"""

import sagemaker
from sagemaker.pytorch import PyTorch
import boto3
import os

# ── Konfigürasyon — sadece burası değiştir ────────────────────
S3_BUCKET       = 'sage-ham10k-eda'                  # senin bucket'ın
S3_DATA_PREFIX  = 'HAM10000'                          # bucket'taki dataset klasörü
S3_OUTPUT       = f's3://{S3_BUCKET}/training-output' # sonuçlar buraya yazılır

INSTANCE_TYPE   = 'ml.g4dn.xlarge'   # quota onaylanınca bunu kullan
# INSTANCE_TYPE = 'ml.g4dn.2xlarge'  # daha hızlı alternatif

USE_SPOT        = True     # %70 indirim — önerilir
MAX_WAIT_SECS   = 86400    # spot için max bekleme: 24 saat
MAX_RUN_SECS    = 86400    # max training süresi: 24 saat

# ── SageMaker session ─────────────────────────────────────────
sess        = sagemaker.Session()
role        = sagemaker.get_execution_role()
region      = boto3.Session().region_name

print(f"Role   : {role}")
print(f"Region : {region}")
print(f"Bucket : {S3_BUCKET}")

# ── Kaynak dosyalar (repo klasörün) ───────────────────────────
# Bu script'i SAGE/ klasörünün içinden çalıştır
SOURCE_DIR = '/home/sagemaker-user/SAGE'

# ── Estimator ─────────────────────────────────────────────────
estimator = PyTorch(
    entry_point        = 'train.py',          # entry point
    source_dir         = SOURCE_DIR,          # tüm repo buradan gider
    role               = role,
    framework_version  = '2.1',
    py_version         = 'py310',
    instance_type      = INSTANCE_TYPE,
    instance_count     = 1,
    output_path        = S3_OUTPUT,
    use_spot_instances = USE_SPOT,
    max_wait           = MAX_WAIT_SECS if USE_SPOT else None,
    max_run            = MAX_RUN_SECS,
    base_job_name      = 'sage-shapfed-ham10k',
    hyperparameters    = {
        'dataset':            'HAM10000',
        'aggregation_method': 'ShapFed',
        'alpha':              1.0,
        'gpu_id':             0,
        'num_clients':        20,
        'num_online_clients': 8,
        'local_epochs':       5,
        'shapley_samples':    10,
        'threshold':          0.95,
        'lambda_u':           1.0,
        'T':                  1.0,
        'mu':                 2,
        'seed':               7,
        'batch_size_local_labeled_fixmatch': 24,
        'batch_size_test':    128,
        'lr_local_training':  0.03,
    },
    environment = {
        'PYTHONPATH': '/opt/ml/code',
    }
)

# ── Dataset S3 input channel ───────────────────────────────────
inputs = {
    'dataset': f's3://{S3_BUCKET}/{S3_DATA_PREFIX}/'
}

# ── Job'u başlat ──────────────────────────────────────────────
print("\n[INFO] Submitting training job...")
print(f"[INFO] Data input  : {inputs['dataset']}")
print(f"[INFO] Output      : {S3_OUTPUT}")
print(f"[INFO] Spot        : {USE_SPOT}")
print(f"[INFO] Instance    : {INSTANCE_TYPE}\n")

estimator.fit(inputs, wait=False)  # wait=False → job submit edilir, notebook bloklanmaz

print(f"\n[INFO] Job submitted: {estimator.latest_training_job.name}")
print(f"[INFO] Logs için: CloudWatch → /aws/sagemaker/TrainingJobs")
print(f"[INFO] Ya da: estimator.logs() ile notebook'tan takip et")
