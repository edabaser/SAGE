import argparse


def args_parser():
    parser = argparse.ArgumentParser(description='SAGE Federated Learning')

    # Donanim
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed',   type=int, default=7)

    # Dataset
    parser.add_argument('--dataset', type=str, default='HAM10000',
                        choices=['HAM10000', 'CIFAR10', 'CIFAR100', 'SVHN', 'CINIC10'])
    parser.add_argument('--path_ham10000', type=str,
                        default='/mnt/sagemaker-nvme/SAGE/data/sage-ham10k-eda')
    parser.add_argument('--path_cifar10',  type=str, default='./data/CIFAR10/')
    parser.add_argument('--path_cifar100', type=str, default='./data/CIFAR100/')
    parser.add_argument('--path_svhn',     type=str, default='./data/SVHN/')
    parser.add_argument('--path_cinic10',  type=str, default='./data/CINIC10/')

    # Model
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet8', 'resnet18'],
                        help='resnet8: orijinal SAGE (CIFAR), resnet18: pretrained (HAM10000)')

    # Federated Learning
    parser.add_argument('--num_clients',        type=int,   default=12)
    parser.add_argument('--num_online_clients', type=int,   default=8)
    parser.add_argument('--num_rounds',         type=int,   default=300)
    parser.add_argument('--num_labeled',        type=int,   default=None,
                        help='Sinif basina sabit etiketli veri sayisi (IPC). '
                             'label_ratio ile birlikte kullanilirsa num_labeled onceliklidir.')
    parser.add_argument('--label_ratio',        type=float, default=0.2,
                        help='Her sinifin kac yuzdesi labeled olacak (0.0-1.0). '
                             'Ornek: 0.1 -> her sinifin %%10u labeled. '
                             'num_labeled verilmisse bu parametre goz ardi edilir.')
    parser.add_argument('--alpha', type=float,  default=1.0,
                        help='Dirichlet alpha (dusuk=heterojen)')

    # Aggregation
    parser.add_argument('--aggregation_method', type=str, default='ShapFed',
                        choices=['FedAvg', 'ShapFed'])
    parser.add_argument('--shapley_samples', type=int, default=10)

    # FixMatch / SSL
    parser.add_argument('--mu',           type=int,   default=2)
    parser.add_argument('--threshold',    type=float, default=0.95)
    parser.add_argument('--lambda_u',     type=float, default=1.0)
    parser.add_argument('--T',            type=float, default=1.0)
    parser.add_argument('--local_epochs', type=int,   default=2)
    parser.add_argument('--kappa',        type=float, default=0.5,
                        help='SAGE CDSC: lambda=exp(-kappa*delta_C). '
                             'Dusuk=local pseudo-labele guvenir, '
                             'Yuksek=buyuk confidence farklarinda global modele kayar. '
                             'Paper default: 0.5')

    # Optimizer
    parser.add_argument('--lr_local_training',        type=float, default=0.001)
    parser.add_argument('--lr_distillation_training', type=float, default=0.01)
    parser.add_argument('--lr_min_ratio',             type=float, default=0.01)

    # Batch Size
    parser.add_argument('--batch_size_local_labeled_fixmatch', type=int, default=32)
    parser.add_argument('--batch_size_local_labeled',          type=int, default=32)
    parser.add_argument('--batch_size_local_unlabeled',        type=int, default=64)
    parser.add_argument('--batch_size_test',                   type=int, default=64)

    # ── Mevcut Ablation Flagleri ─────────────────────────────
    parser.add_argument('--use_focal_loss',       action='store_true', default=False,
                        help='Focal Loss (kapali=CrossEntropy)')
    parser.add_argument('--use_stfl',             action='store_true', default=False,
                        help='STFL dinamik threshold')
    parser.add_argument('--stfl_beta',            type=float, default=0.6)
    parser.add_argument('--use_weighted_sampler', action='store_true', default=False,
                        help='WeightedRandomSampler')
    parser.add_argument('--use_medical_augment',  action='store_true', default=False,
                        help='Medikal-guvenli augmentation (renk bozucu oplar cikarildi)')
    parser.add_argument('--use_cosine_lr',        action='store_true', default=False,
                        help='CosineAnnealingLR')
    parser.add_argument('--use_groupnorm',        action='store_true', default=False,
                        help='ResNet18 icin BN->GN donusumu')
    parser.add_argument('--group_norm_num_groups', type=int, default=16)

    # ── Yeni ShapFed Guclendirme Flagleri ────────────────────
    # V1: Global Model EMA ile eval — zigzag azalir
    parser.add_argument('--use_ema_eval', action='store_true', default=False,
                        help='[V1] EMA modeli ile eval (zigzag azalir)')
    parser.add_argument('--ema_decay', type=float, default=0.95,
                        help='EMA decay katsayisi (0.90-0.99)')

    # V2: Shapley agirliklarinda EMA yumusatma — ShapFed != FedAvg icin
    parser.add_argument('--use_shapley_ema', action='store_true', default=False,
                        help='[V2] Shapley agirliklarinda EMA (erken round gurultusunu azaltir)')
    parser.add_argument('--shapley_ema_decay', type=float, default=0.7,
                        help='Shapley EMA decay (0.5-0.9)')

    # V3: Personalized Broadcast — orijinal ShapFed paper mekanizmasi
    parser.add_argument('--use_personalized_init', action='store_true', default=False,
                        help='[V3] Personalized broadcast (ShapFed paper orijinal mekanizmasi)')
    parser.add_argument('--personalization_strength', type=float, default=1.0,
                        help='Shapley skoru * strength = global model karisim orani [0,1]')

    # AWS
    parser.add_argument('--checkpoint_dir', type=str,
                        default='/home/sagemaker-user/SAGE/checkpoints')
    parser.add_argument('--s3_bucket', type=str, default='sage-ham10k-eda')

    args = parser.parse_args()
    return args
