# import argparse
# import os

# def args_parser():
#     parser = argparse.ArgumentParser()
    
#     # --- Donanim ve Veriseti ---
#     parser.add_argument('--gpu_id', type=int, default=0)
#     parser.add_argument('--dataset', type=str, default='HAM10000') 
#     parser.add_argument('--num_clients', type=int, default=20)
#     parser.add_argument('--num_online_clients', type=int, default=8)  
#     parser.add_argument('--num_labeled', type=int, default=1000, help='number of labeled data')

#     # --- SAGE/FixMatch Parametreleri ---
#     parser.add_argument('--mu', default=2, type=int)
#     parser.add_argument('--alpha', type=float, default=0.1)
#     parser.add_argument('--threshold', default=0.95, type=float)
#     parser.add_argument('--lambda_u', default=1, type=float)
#     parser.add_argument('--kappa', default=0.5, type=float)
#     parser.add_argument('--T', default=1, type=float)

#     # --- Egitim Parametreleri ---
#     parser.add_argument('--local_epochs', type=int, default=2) # 5--> 2
#     parser.add_argument('--batch_size_local_labeled_fixmatch', type=int, default=128)
#     parser.add_argument('--batch_size_local_labeled', type=int, default=128)  
#     parser.add_argument('--batch_size_local_unlabeled', type=int, default=128)
#     parser.add_argument('--batch_size_test', type=int, default=128)
#     parser.add_argument('--lr_local_training', type=float, default=0.1)
#     parser.add_argument('--lr_distillation_training', type=float, default=0.01)

#     parser.add_argument('--group_norm_num_groups', type=int, default=16, help="Number of groups for GN")

#     # --- Aggregation ve Shapley ---
#     parser.add_argument('--aggregation_method', type=str, default='ShapFed') 
#     parser.add_argument('--shapley_samples', type=int, default=10)

#     # --- AWS SageMaker / S3 Yollari ---
#     # Not: Klasor yollarinin sonundaki / isaretine dikkat
#     parser.add_argument('--checkpoint_dir', type=str, default='/home/sagemaker-user/SAGE/checkpoints') 
#     parser.add_argument('--path_ham10000', type=str, default='/mnt/sagemaker-nvme/SAGE/data/sage-ham10k-eda')
#     parser.add_argument('--s3_bucket', type=str, default='sage-ham10k-eda')
    
#     # --- Diger ---
#     parser.add_argument('--path_cifar10', type=str, default='./data/CIFAR10/')
#     parser.add_argument('--path_cifar100', type=str, default='./data/CIFAR100/')
#     parser.add_argument('--path_svhn', type=str, default='./data/SVHN/')
#     parser.add_argument('--path_cinic10', type=str, default='./data/CINIC10/')
#     parser.add_argument('--seed', type=int, default=7)
    
#     args = parser.parse_args()
#     return args

import argparse
import os

def args_parser():
    parser = argparse.ArgumentParser()
    
    # --- Donanım ve Veriseti ---
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='HAM10000') 
    
    # YENİ STRATEJİ: Toplam 12 Client, her round 8'i aktif
    parser.add_argument('--num_clients', type=int, default=12)
    parser.add_argument('--num_online_clients', type=int, default=8)  
    parser.add_argument('--num_labeled', type=int, default=100, help='Çok az etiketli veri senaryosu (100)')

    # --- SAGE/FixMatch Parametreleri ---
    parser.add_argument('--mu', default=2, type=int)
    parser.add_argument('--alpha', type=float, default=1.0) # Dirichlet Alpha (1.0 daha homojen dağıtır)
    
    # STFL Kullanacağımız için bu Threshold artık başlangıç/maksimum eşik olacak
    parser.add_argument('--threshold', default=0.85, type=float) 
    parser.add_argument('--lambda_u', default=1, type=float)
    parser.add_argument('--kappa', default=0.5, type=float)
    parser.add_argument('--T', default=1, type=float)

    # --- Eğitim Parametreleri ---
    parser.add_argument('--local_epochs', type=int, default=2) 
    parser.add_argument('--batch_size_local_labeled_fixmatch', type=int, default=64) # 128 RAM'i zorlayabilir, 64 daha güvenli
    parser.add_argument('--batch_size_local_labeled', type=int, default=64)  
    parser.add_argument('--batch_size_local_unlabeled', type=int, default=64)
    parser.add_argument('--batch_size_test', type=int, default=64)
    parser.add_argument('--lr_local_training', type=float, default=0.001) # Pretrained model kullandığımız için LR düşürdük! (0.1 çok yüksek)
    parser.add_argument('--lr_distillation_training', type=float, default=0.01)

    parser.add_argument('--group_norm_num_groups', type=int, default=16, help="Number of groups for GN")

    # --- Aggregation ve Shapley ---
    parser.add_argument('--aggregation_method', type=str, default='ShapFed') 
    parser.add_argument('--shapley_samples', type=int, default=10)

    # --- AWS SageMaker / S3 Yolları ---
    parser.add_argument('--checkpoint_dir', type=str, default='/home/sagemaker-user/SAGE/checkpoints') 
    parser.add_argument('--path_ham10000', type=str, default='/mnt/sagemaker-nvme/SAGE/data/sage-ham10k-eda')
    parser.add_argument('--path_cifar10', type=str, default='./data', help="CIFAR10 verisinin indirileceği/bulunduğu kök dizin")
    parser.add_argument('--s3_bucket', type=str, default='sage-ham10k-eda')
    
    # --- Diğer ---
    parser.add_argument('--seed', type=int, default=7)
    
    
    args = parser.parse_args()
    return args
