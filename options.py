# -*-coding:utf-8-*-
import argparse
import os


def args_parser():
    parser = argparse.ArgumentParser()
    path_dir = os.path.dirname(__file__)

    # --- Hardware and Dataset Configuration ---
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='HAM10000',
                        help='Supported datasets: CIFAR10 / CIFAR100 / SVHN / CINIC10 / HAM10000 / BloodMNIST') 
    parser.add_argument('--num_clients', type=int, default=20,
                        help='Total number of clients in the federated learning setup')
    parser.add_argument('--num_online_clients', type=int, default=8,
                        help='Number of clients participating in each communication round')

    # --- Semi-Supervised Learning (SAGE/FixMatch) Parameters ---
    parser.add_argument('--mu', default=2, type=int,
                        help='Number of augmentations for unlabeled samples (unlabeled batch size = labeled batch size * mu)')
    parser.add_argument('--alpha', type=float, default=1,
                        help='Dirichlet distribution parameter for Non-IID data partitioning')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='Pseudo-label threshold for high confidence selection')
    parser.add_argument('--lambda_u', default=1, type=float,
                        help='Coefficient of unlabeled loss (Lu)')
    parser.add_argument('--kappa', default=0.5, type=float,
                        help='Hyperparameter for controlling sensitivity to confidence discrepancy in CDSC (SAGE)')
    parser.add_argument('--T', default=1, type=float,
                        help='Temperature of pseudo-labeling softmax')

    # --- Training Parameters ---
    parser.add_argument('--local_epochs', type=int, default=5,
                        help='Local training epochs for each client')
    parser.add_argument('--batch_size_local_labeled_fixmatch', type=int, default=128)
    parser.add_argument('--batch_size_local_labeled', type=int, default=128)
    parser.add_argument('--batch_size_local_unlabeled', type=int, default=128)
    parser.add_argument('--batch_size_test', type=int, default=512)
    parser.add_argument('--lr_local_training', type=float, default=0.1)
    parser.add_argument('--lr_distillation_training', type=float, default=0.1)

    # --- Dataset Paths ---
    parser.add_argument('--path_cifar10', type=str, default=os.path.join(path_dir, 'data/CIFAR10/'))
    parser.add_argument('--path_cifar100', type=str, default=os.path.join(path_dir, 'data/CIFAR100/'))
    parser.add_argument('--path_svhn', type=str, default=os.path.join(path_dir, 'data/SVHN/'))
    parser.add_argument('--path_cinic10', type=str, default=os.path.join(path_dir, 'data/CINIC10/'))
    parser.add_argument('--path_ham10000', type=str, default=os.path.join(path_dir, 'data/HAM10000/')) # NEW: Path for HAM10000
    parser.add_argument('--path_bloodmnist', type=str, default=os.path.join(path_dir, 'data/BloodMNIST/')) # NEW: Path for BloodMNIST

    # --- Shapley-Driven FL (SDFL) Argümanları ---
    parser.add_argument('--aggregation_method', type=str, default='SDFL_SAGE', # Default set to the enhanced method
                        help='Aggregation method: SAGE (FedAvg), SDFL_SAGE (Shapley-Driven)') 
    parser.add_argument('--shapley_samples', type=int, default=10,
                        help='Number of Monte Carlo samples for Shapley estimation.')

    # --- Ablation Study & Baselines (Keeping for completeness) ---
    parser.add_argument('--SAGE_fixed_p', default=0.5, type=float,
                        help='Fixed parameter used in SAGE ablation study instead of dynamic adjustment')
    parser.add_argument('--ablation_kappa', default=2, type=float,
                        help='Hyperparameter for controlling sensitivity of exponential decay in SAGE ablation study')
    parser.add_argument('--seed', type=int, default=7)
    
    # FedProx
    parser.add_argument('--lambda_prox', default=0.001, type=float,
                        help='coefficient of FedProx')
    # FedLabel, FreeMatch, FedMatch args... (omitted here but assumed to remain)

    parser.add_argument('--num_epochs_label_distillation', type=int, default=50)
    parser.add_argument('--num_epochs_unlabel_distillation', type=int, default=50)
    parser.add_argument('--batch_size_label_distillation', type=int, default=128)
    parser.add_argument('--batch_size_unlabel_distillation', type=int, default=128)
    
    args = parser.parse_args()
    return args
