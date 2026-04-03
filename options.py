# import argparse
# import os

# def args_parser():
#     parser = argparse.ArgumentParser()
#     path_dir = os.path.dirname(__file__)

#     # --- Hardware and Dataset Configuration ---
#     parser.add_argument('--gpu_id', type=int, default=0)
#     parser.add_argument('--dataset', type=str, default='CIFAR10',
#                         help='Supported: CIFAR10, CIFAR100, SVHN, CINIC10, HAM10000') 
#     parser.add_argument('--num_clients', type=int, default=20,
#                         help='Total number of clients')
#     parser.add_argument('--num_online_clients', type=int, default=8,
#                         help='Number of clients participating in each round')

#     # --- SAGE/FixMatch Parameters ---
#     parser.add_argument('--mu', default=2, type=int, help='Augmentations factor for unlabeled')
#     parser.add_argument('--alpha', type=float, default=1, help='Dirichlet distribution parameter (Non-IID)')
#     parser.add_argument('--threshold', default=0.95, type=float, help='Pseudo-label threshold')
#     parser.add_argument('--lambda_u', default=1, type=float, help='Coefficient of unlabeled loss')
#     parser.add_argument('--kappa', default=0.5, type=float, help='Hyperparameter for CDSC (SAGE)')
#     parser.add_argument('--T', default=1, type=float, help='Temperature')

#     # --- Training Parameters ---
#     parser.add_argument('--local_epochs', type=int, default=5)
#     parser.add_argument('--batch_size_local_labeled_fixmatch', type=int, default=24) # 64 -> 24 (RAM tasarrufu)
#     parser.add_argument('--batch_size_local_labeled', type=int, default=24)  # 24-->32
#     parser.add_argument('--batch_size_local_unlabeled', type=int, default=24)
#     parser.add_argument('--batch_size_test', type=int, default=128)
#     parser.add_argument('--lr_local_training', type=float, default=0.03)
#     parser.add_argument('--lr_distillation_training', type=float, default=0.01)

#     # --- Checkpoint & Aggregation ---
#     parser.add_argument('--checkpoint_dir', type=str, default='/content/drive/MyDrive/Colab Notebooks/EE 401/SAGE-master.v1/Checkpoints',
#                         help='Directory to save/load model checkpoints.') 
    
#     # Aggregation Method: 'SAGE' (Standard FedAvg) or 'ShapFed'
#     parser.add_argument('--aggregation_method', type=str, default='ShapFed',
#                         help='Aggregation method: SAGE (FedAvg) or ShapFed') 
    
#     # ShapFed için Monte Carlo örnekleme sayısı (Hız/Doğruluk dengesi için)
#     parser.add_argument('--shapley_samples', type=int, default=10,
#                         help='Number of Monte Carlo samples for Shapley estimation.')

#     # --- Dataset Paths ---
    
#     parser.add_argument('--path_cifar10', type=str, default='./data/CIFAR10/')
#     parser.add_argument('--path_cifar100', type=str, default='./data/CIFAR100/')
#     parser.add_argument('--path_svhn', type=str, default='./data/SVHN/')
#     parser.add_argument('--path_cinic10', type=str, default='./data/CINIC10/')
    
#     parser.add_argument('--path_ham10000', type=str, default='./data/HAM10000/')
    
#     parser.add_argument('--seed', type=int, default=7, help='random seed')
    
#     args = parser.parse_args()
#     return args
    
    # import argparse
    # import os
    
    # def args_parser():
    #     parser = argparse.ArgumentParser()
    #     path_dir = os.path.dirname(__file__)
    
    #     # --- Hardware and Dataset Configuration ---
    #     parser.add_argument('--gpu_id', type=int, default=0)
    #     parser.add_argument('--dataset', type=str, default='HAM10000',  # Varsayılanı HAM10000 yaptık
    #                         help='Supported: CIFAR10, CIFAR100, SVHN, CINIC10, HAM10000') 
    #     parser.add_argument('--num_clients', type=int, default=20,
    #                         help='Total number of clients')
    #     parser.add_argument('--num_online_clients', type=int, default=8,
    #                         help='Number of clients participating in each round')
    
    #     # --- SAGE/FixMatch Parameters ---
    #     parser.add_argument('--mu', default=2, type=int, help='Augmentations factor for unlabeled')
    #     parser.add_argument('--alpha', type=float, default=1.0, help='Dirichlet distribution parameter (Non-IID)')
    #     parser.add_argument('--threshold', default=0.95, type=float, help='Pseudo-label threshold')
    #     parser.add_argument('--lambda_u', default=1, type=float, help='Coefficient of unlabeled loss')
    #     parser.add_argument('--kappa', default=0.5, type=float, help='Hyperparameter for CDSC (SAGE)')
    #     parser.add_argument('--T', default=1, type=float, help='Temperature')
    
    #     # --- Training Parameters ---
    #     parser.add_argument('--local_epochs', type=int, default=5)
    #     parser.add_argument('--batch_size_local_labeled_fixmatch', type=int, default=24) 
    #     parser.add_argument('--batch_size_local_labeled', type=int, default=24)  
    #     parser.add_argument('--batch_size_local_unlabeled', type=int, default=24)
    #     parser.add_argument('--batch_size_test', type=int, default=128)
    #     parser.add_argument('--lr_local_training', type=float, default=0.03)
    #     parser.add_argument('--lr_distillation_training', type=float, default=0.01)
    
    #     # --- Aggregation Method ---
    #     parser.add_argument('--aggregation_method', type=str, default='ShapFed',
    #                         help='Aggregation method: SAGE (FedAvg) or ShapFed') 
        
    #     # ShapFed için Monte Carlo örnekleme sayısı
    #     parser.add_argument('--shapley_samples', type=int, default=10,
    #                         help='Number of Monte Carlo samples for Shapley estimation.')
    
    #     # --- AWS SageMaker Snipets ---
    #     parser.add_argument('--checkpoint_dir', type=str, 
    #                         default='/home/sagemaker-user/SageMaker/checkpoints',
    #                         help='Local directory to save checkpoints before S3 sync') 
        
    #     parser.add_argument('--s3_bucket', type=str, default='sage-ham10k-eda',
    #                         help='The name of your S3 bucket')
        
    #     # parser.add_argument('--path_ham10000', type=str, 
    #                         default='/home/sagemaker-user/SageMaker/data/HAM10000/',
    #                         help='Local path where S3 data will be synced')
      
    #     parser.add_argument('--path_ham10000', type=str, default='/home/sagemaker-user/SAGE/data/HAM10000/')
    
    
    #     # --- Dataset Paths (Digerleri) ---
    #     parser.add_argument('--path_cifar10', type=str, default='./data/CIFAR10/')
    #     parser.add_argument('--path_cifar100', type=str, default='./data/CIFAR100/')
    #     parser.add_argument('--path_svhn', type=str, default='./data/SVHN/')
    #     parser.add_argument('--path_cinic10', type=str, default='./data/CINIC10/')
        
    #     parser.add_argument('--seed', type=int, default=7, help='random seed')
        
    #     args = parser.parse_args()
    #     return args

import argparse
import os

def args_parser():
    parser = argparse.ArgumentParser()
    
    # --- Donanim ve Veriseti ---
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='HAM10000') 
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--num_online_clients', type=int, default=8)

    # --- SAGE/FixMatch Parametreleri ---
    parser.add_argument('--mu', default=2, type=int)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--threshold', default=0.95, type=float)
    parser.add_argument('--lambda_u', default=1, type=float)
    parser.add_argument('--kappa', default=0.5, type=float)
    parser.add_argument('--T', default=1, type=float)

    # --- Egitim Parametreleri ---
    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--batch_size_local_labeled_fixmatch', type=int, default=16) #24-->16
    parser.add_argument('--batch_size_local_labeled', type=int, default=16)  
    parser.add_argument('--batch_size_local_unlabeled', type=int, default=16)
    parser.add_argument('--batch_size_test', type=int, default=128)
    parser.add_argument('--lr_local_training', type=float, default=0.03)
    parser.add_argument('--lr_distillation_training', type=float, default=0.01)

    # --- Aggregation ve Shapley ---
    parser.add_argument('--aggregation_method', type=str, default='ShapFed') 
    parser.add_argument('--shapley_samples', type=int, default=10)

    # --- AWS SageMaker / S3 Yollari ---
    # Not: Klasor yollarinin sonundaki / isaretine dikkat
    parser.add_argument('--checkpoint_dir', type=str, default='/home/sagemaker-user/SAGE/checkpoints') 
    parser.add_argument('--path_ham10000', type=str, default='/mnt/sagemaker-nvme/SAGE/data/sage-ham10k-eda')
    parser.add_argument('--s3_bucket', type=str, default='sage-ham10k-eda')
    
    # --- Diger ---
    parser.add_argument('--path_cifar10', type=str, default='./data/CIFAR10/')
    parser.add_argument('--path_cifar100', type=str, default='./data/CIFAR100/')
    parser.add_argument('--path_svhn', type=str, default='./data/SVHN/')
    parser.add_argument('--path_cinic10', type=str, default='./data/CINIC10/')
    parser.add_argument('--seed', type=int, default=7)
    
    args = parser.parse_args()
    return args
