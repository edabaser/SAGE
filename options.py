# # -*-coding:utf-8-*-
# import argparse
# import os


# def args_parser():
#     parser = argparse.ArgumentParser()
#     path_dir = os.path.dirname(__file__)

#     # --- Hardware and Dataset Configuration ---
#     parser.add_argument('--gpu_id', type=int, default=0)
#     parser.add_argument('--dataset', type=str, default='HAM10000',
#                         help='Supported datasets: CIFAR10 / CIFAR100 / SVHN / CINIC10 / HAM10000') 
#     parser.add_argument('--num_clients', type=int, default=20,
#                         help='Total number of clients in the federated learning setup')
#     parser.add_argument('--num_online_clients', type=int, default=8,
#                         help='Number of clients participating in each communication round')

#     # --- Semi-Supervised Learning (SAGE/FixMatch) Parameters ---
#     parser.add_argument('--mu', default=2, type=int, help='Number of augmentations for unlabeled samples')
#     parser.add_argument('--alpha', type=float, default=1, help='Dirichlet distribution parameter for Non-IID data')
#     parser.add_argument('--threshold', default=0.95, type=float, help='Pseudo-label threshold')
#     parser.add_argument('--lambda_u', default=1, type=float, help='Coefficient of unlabeled loss (Lu)')
#     parser.add_argument('--kappa', default=0.5, type=float, help='Hyperparameter for CDSC (SAGE)')
#     parser.add_argument('--T', default=1, type=float, help='Temperature of pseudo-labeling softmax')

#     # --- Training Parameters ---
#     parser.add_argument('--local_epochs', type=int, default=5, help='Local training epochs for each client')
#     parser.add_argument('--batch_size_local_labeled_fixmatch', type=int, default=128)
#     parser.add_argument('--batch_size_local_labeled', type=int, default=128)
#     parser.add_argument('--batch_size_local_unlabeled', type=int, default=128)
#     parser.add_argument('--batch_size_test', type=int, default=512)
#     parser.add_argument('--lr_local_training', type=float, default=0.1)
#     parser.add_argument('--lr_distillation_training', type=float, default=0.1)

#     # --- Checkpoint & Aggregation ---
#     parser.add_argument('--checkpoint_dir', type=str, default=os.path.join(path_dir, 'checkpoints'),
#                         help='Directory to save/load model checkpoints.') # FIX: checkpoint_dir added
#     parser.add_argument('--aggregation_method', type=str, default='SDFL_SAGE',
#                         help='Aggregation method: SAGE (FedAvg), SDFL_SAGE (Shapley-Driven)') 
#     parser.add_argument('--shapley_samples', type=int, default=10,
#                         help='Number of Monte Carlo samples for Shapley estimation.')

#     # --- Dataset Paths ---
#     parser.add_argument('--path_cifar10', type=str, default=os.path.join(path_dir, 'data/CIFAR10/'))
#     parser.add_argument('--path_cifar100', type=str, default=os.path.join(path_dir, 'data/CIFAR100/'))
#     parser.add_argument('--path_svhn', type=str, default=os.path.join(path_dir, 'data/SVHN/'))
#     parser.add_argument('--path_cinic10', type=str, default=os.path.join(path_dir, 'data/CINIC10/'))
#     parser.add_argument('--path_ham10000', type=str, default=os.path.join(path_dir, 'data/HAM10000/')) # HAM10000 Path
    
#     # --- Ablation Study & Baselines ---
#     parser.add_argument('--ablation', type=str, default='0')
#     parser.add_argument('--seed', type=int, default=7, help='random seed')
    
#     args = parser.parse_args()
#     return args


# options.py
import argparse
import os

def args_parser():
    parser = argparse.ArgumentParser()
    path_dir = os.path.dirname(__file__)

    # --- Hardware and Dataset Configuration ---
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='Supported: CIFAR10, CIFAR100, SVHN, CINIC10, HAM10000') 
    parser.add_argument('--num_clients', type=int, default=100,
                        help='Total number of clients')
    parser.add_argument('--num_online_clients', type=int, default=10,
                        help='Number of clients participating in each round')

    # --- SAGE/FixMatch Parameters ---
    parser.add_argument('--mu', default=2, type=int, help='Augmentations factor for unlabeled')
    parser.add_argument('--alpha', type=float, default=1, help='Dirichlet distribution parameter (Non-IID)')
    parser.add_argument('--threshold', default=0.95, type=float, help='Pseudo-label threshold')
    parser.add_argument('--lambda_u', default=1, type=float, help='Coefficient of unlabeled loss')
    parser.add_argument('--kappa', default=0.5, type=float, help='Hyperparameter for CDSC (SAGE)')
    parser.add_argument('--T', default=1, type=float, help='Temperature')

    # --- Training Parameters ---
    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--batch_size_local_labeled_fixmatch', type=int, default=24) # 64 -> 24 (RAM tasarrufu)
    parser.add_argument('--batch_size_local_labeled', type=int, default=24)
    parser.add_argument('--batch_size_local_unlabeled', type=int, default=24)
    parser.add_argument('--batch_size_test', type=int, default=128)
    parser.add_argument('--lr_local_training', type=float, default=0.03)
    parser.add_argument('--lr_distillation_training', type=float, default=0.01)

    # --- Checkpoint & Aggregation (YENİ EKLENEN KISIMLAR) ---
    parser.add_argument('--checkpoint_dir', type=str, default='/content/drive/MyDrive/Colab Notebooks/EE 401/SAGE-master.v1/Checkpoints',
                        help='Directory to save/load model checkpoints.') 
    
    # Aggregation Method: 'SAGE' (Standart FedAvg) veya 'ShapFed'
    parser.add_argument('--aggregation_method', type=str, default='ShapFed',
                        help='Aggregation method: SAGE (FedAvg) or ShapFed') 
    
    # ShapFed için Monte Carlo örnekleme sayısı (Hız/Doğruluk dengesi için)
    parser.add_argument('--shapley_samples', type=int, default=10,
                        help='Number of Monte Carlo samples for Shapley estimation.')

    # --- Dataset Paths ---
    parser.add_argument('--path_cifar10', type=str, default='./data/CIFAR10/')
    parser.add_argument('--path_cifar100', type=str, default='./data/CIFAR100/')
    parser.add_argument('--path_svhn', type=str, default='./data/SVHN/')
    parser.add_argument('--path_cinic10', type=str, default='./data/CINIC10/')
    parser.add_argument('--path_ham10000', type=str, default='./data/HAM10000/')
    
    parser.add_argument('--seed', type=int, default=7, help='random seed')
    
    args = parser.parse_args()
    return args
