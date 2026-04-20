# import numpy as np
# from torch.utils.data.dataset import Dataset
# import torch
# from PIL import Image
# from torchvision import transforms
# from .randaugment import RandAugmentMC
# from Dataset.sample_dirichlet import clients_indices, clients_indices_unlabel
# from collections import Counter

# def classify_label(dataset, num_classes: int):
#     list_label2indices = [[] for _ in range(num_classes)]
#     targets = getattr(dataset, 'targets', None)
#     if targets is None and hasattr(dataset, 'base_dataset'):
#         targets = dataset.targets 
#     for idx, label in enumerate(targets):
#         list_label2indices[label].append(idx)
#     return list_label2indices

# def show_clients_data_distribution(dataset, clients_indices_labeled, clients_indices_unlabeled, num_classes):
#     dict_per_client_labeled = []
#     for client, indices in enumerate(zip(clients_indices_labeled, clients_indices_unlabeled)):
#         nums_data_labeled = [0 for _ in range(num_classes)]
#         nums_data_unlabeled = [0 for _ in range(num_classes)]
#         idx_labeled, idx_unlabeled = indices
#         for idx in idx_labeled:
#             label = dataset[idx][1]
#             nums_data_labeled[label] += 1
#         dict_per_client_labeled.append(nums_data_labeled)
#         for idx in idx_unlabeled:
#             label = dataset[idx][1]
#             nums_data_unlabeled[label] += 1
#     return dict_per_client_labeled

# def partition_train(list_label2indices: list, ipc):
#     list_label2indices_labeled = []
#     list_label2indices_unlabeled = []
#     for indices in list_label2indices:
#         idx_shuffle = np.random.permutation(indices)
#         list_label2indices_labeled.append(idx_shuffle[:ipc])
#         list_label2indices_unlabeled.append(idx_shuffle[ipc:])
#     return list_label2indices_labeled, list_label2indices_unlabeled

# class Indices2Dataset_labeled(Dataset):
#     def __init__(self, dataset, dataset_name='CIFAR10'):
#         self.dataset = dataset
#         self.dataset_name = dataset_name
#         self.indices = None

#     def load(self, indices: list):
#         self.indices = indices
#         if self.dataset_name == 'HAM10000':
#             self.client_dataset = [self.dataset[i] for i in indices]
#             # Çarpanı maksimum 3 veya 5'te tut (Iteration Stop hatasını engellemek için yeterli)
#             self.client_dataset *= 3

#     #eski ctastrohic forgetting code
#         # self.client_dataset = [self.dataset[i] for i in indices]
        
#         # # Orijinal SAGE Hack: Labeled veriyi çoğalt ki Iteration Stop hatası vermesin
#         # if self.dataset_name == 'HAM10000':
#         #     self.client_dataset *= 10 # Medikal veri çok ağır, fazla çarpmıyoruz
#         # else:
#         #     self.client_dataset *= 2000 # ORİJİNAL SAGE ÇARPANI
    
# # # yeni - Inverse Frequency Balancing

# #         if self.dataset_name == 'HAM10000':
            
# #             # 1. Bu istemciye atanan verilerin etiketlerini bul
# #             targets = [self.dataset.targets[i] for i in indices]
# #             class_counts = Counter(targets)
            
# #             # Eğer istemcide hiç veri yoksa çökmemesi için güvenlik
# #             if not class_counts:
# #                 self.client_dataset = []
# #                 return
                
# #             max_count = max(class_counts.values())
            
# #             self.client_dataset = []
# #             for idx, target in zip(indices, targets):
# #                 # 2. Dinamik Çarpan: Nadir sınıf çok çoğaltılır, baskın sınıf (nv) az çoğaltılır.
# #                 # Örn: nv 100 tane, vasc 5 taneyse -> nv çarpanı 1, vasc çarpanı 20 olur.
# #                 # Base multiplier (örneğin 3) ekleyerek Iteration Stop hatasını önlüyoruz.
# #                 base_multiplier = 2 
# #                 weight = max(1, int(max_count / class_counts[target])) * base_multiplier
# #                 self.client_dataset.extend([self.dataset[idx]] * weight)
                
# #         else:
# #             self.client_dataset = [self.dataset[i] for i in indices]
# #             self.client_dataset *= 2000
# # #yeni sonu
    
#     def __getitem__(self, idx):
#         image, label = self.client_dataset[idx]
        
#         if self.dataset_name == 'HAM10000':
#             if torch.is_tensor(image): image = transforms.ToPILImage()(image)
#             trans = transforms.Compose([
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomVerticalFlip(),
#                 transforms.RandomCrop(size=224, padding=28, padding_mode='reflect'),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=(0.763, 0.545, 0.570), std=(0.140, 0.152, 0.169))
#             ])
#             return trans(image), label
#         else:
#             # STRICT ORIGINAL SAGE LOGIC FOR CIFAR
#             trans = transforms.Compose([
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect'),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
#             ])
#             return trans(image), label

#     def __len__(self):
#         # Orijinal FedAvg ağırlığını korumak için çoğaltılmış uzunluğu dönmek ZORUNDAYIZ
#         return len(self.client_dataset)

# class Indices2Dataset_unlabeled_fixmatch(Dataset):
#     def __init__(self, dataset, dataset_name='CIFAR10'):
#         self.dataset = dataset
#         self.dataset_name = dataset_name
#         self.indices = None

#     def load(self, indices: list):
#         self.indices = indices
#         if self.dataset_name == 'HAM10000':
#             self.client_dataset = [self.dataset[i] for i in self.indices]
#             self.client_dataset *= 3
#             self.client_dataset_len = len(self.client_dataset)
#         # self.client_dataset = [self.dataset[i] for i in self.indices]
#         # self.client_dataset_len = len(self.client_dataset)
        
#         # if self.dataset_name == 'HAM10000':
#         #     self.client_dataset *= 5 
#         # else:
#         #     self.client_dataset *= 50 # ORİJİNAL SAGE ÇARPANI

#     # #Inverse Frequenct-y Balancing
#     #     if self.dataset_name == 'HAM10000':
#     #         targets = [self.dataset.targets[i] for i in self.indices]
#     #         class_counts = Counter(targets)
            
#     #         if not class_counts:
#     #             self.client_dataset = []
#     #             self.client_dataset_len = 0
#     #             return
                
#     #         max_count = max(class_counts.values())
#     #         self.client_dataset = []
            
#     #         for idx, target in zip(self.indices, targets):
#     #             # Unlabeled için base_multiplier'ı 2 yapıyoruz çünkü FixMatch 
#     #             # zaten unlabeled veriyi args.mu (genelde 2 veya 7) ile çarpıp çekiyor.
#     #             base_multiplier = 2
#     #             weight = max(1, int(max_count / class_counts[target])) * base_multiplier
#     #             self.client_dataset.extend([self.dataset[idx]] * weight)
                
#     #         self.client_dataset_len = len(self.client_dataset)
#         else:
#             # STRICT ORIGINAL SAGE LOGIC FOR CIFAR
#             self.client_dataset = [self.dataset[i] for i in self.indices]
#             self.client_dataset *= 50
#             self.client_dataset_len = len(self.client_dataset)



    

#     def fixmatch(self, image):
#         if self.dataset_name == 'HAM10000':
#             if torch.is_tensor(image): image = transforms.ToPILImage()(image)
#             weak = transforms.Compose([
#                 transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
#                 transforms.RandomCrop(size=224, padding=28, padding_mode='reflect'),
#                 transforms.ToTensor(), transforms.Normalize((0.763, 0.545, 0.570), (0.140, 0.152, 0.169))
#             ])
#             strong = transforms.Compose([
#                 transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
#                 transforms.RandomCrop(size=224, padding=28, padding_mode='reflect'),
#                 RandAugmentMC(n=2, m=10),
#                 transforms.ToTensor(), transforms.Normalize((0.763, 0.545, 0.570), (0.140, 0.152, 0.169))
#             ])
#             return weak(image), strong(image)
#         else:
#             # STRICT ORIGINAL SAGE LOGIC FOR CIFAR
#             weak = transforms.Compose([
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect'),
#                 transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
#             ])
#             strong = transforms.Compose([
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
#                 RandAugmentMC(n=2, m=10),
#                 transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
#             ])
#             return weak(image), strong(image)

#     def __getitem__(self, idx):
#         image, label = self.client_dataset[idx]
#         image1, image2 = self.fixmatch(image)
#         return image1, image2, label

#     def __len__(self):
#         # Orijinal SAGE: Client Drift olmasın diye iterasyonu unmultiplied boyutla sınırlar!
#         return self.client_dataset_len



# pretrained model için temiz dataset.py

import numpy as np
from torch.utils.data.dataset import Dataset
import torch
from PIL import Image
from torchvision import transforms
from .randaugment import RandAugmentMC
from Dataset.sample_dirichlet import clients_indices, clients_indices_unlabel
from collections import Counter

def classify_label(dataset, num_classes: int):
    list_label2indices = [[] for _ in range(num_classes)]
    targets = getattr(dataset, 'targets', None)
    if targets is None and hasattr(dataset, 'base_dataset'):
        targets = dataset.targets 
    for idx, label in enumerate(targets):
        list_label2indices[label].append(idx)
    return list_label2indices

def show_clients_data_distribution(dataset, clients_indices_labeled, clients_indices_unlabeled, num_classes):
    dict_per_client_labeled = []
    for client, indices in enumerate(zip(clients_indices_labeled, clients_indices_unlabeled)):
        nums_data_labeled = [0 for _ in range(num_classes)]
        nums_data_unlabeled = [0 for _ in range(num_classes)]
        idx_labeled, idx_unlabeled = indices
        for idx in idx_labeled:
            label = dataset[idx][1]
            nums_data_labeled[label] += 1
        dict_per_client_labeled.append(nums_data_labeled)
        for idx in idx_unlabeled:
            label = dataset[idx][1]
            nums_data_unlabeled[label] += 1
    return dict_per_client_labeled

def partition_train(list_label2indices: list, ipc):
    list_label2indices_labeled = []
    list_label2indices_unlabeled = []
    for indices in list_label2indices:
        idx_shuffle = np.random.permutation(indices)
        list_label2indices_labeled.append(idx_shuffle[:ipc])
        list_label2indices_unlabeled.append(idx_shuffle[ipc:])
    return list_label2indices_labeled, list_label2indices_unlabeled

class Indices2Dataset_labeled(Dataset):
    def __init__(self, dataset, dataset_name='CIFAR10'):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.indices = None

    def load(self, indices: list):
        self.indices = indices
        # Önce listeyi oluşturuyoruz
        self.client_dataset = [self.dataset[i] for i in indices]
        
        # Sonra veri setine göre çoğaltıyoruz (Iteration Stop engellemek için)
        if self.dataset_name == 'HAM10000':
            self.client_dataset *= 3
        else:
            self.client_dataset *= 2000 # Orijinal SAGE CIFAR çarpanı
            
    def __getitem__(self, idx):
        image, label = self.client_dataset[idx]
        
        if self.dataset_name == 'HAM10000':
            if torch.is_tensor(image): image = transforms.ToPILImage()(image)
            trans = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomCrop(size=224, padding=28, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.763, 0.545, 0.570), std=(0.140, 0.152, 0.169))
            ])
            return trans(image), label
        else:
            # STRICT ORIGINAL SAGE LOGIC FOR CIFAR
            trans = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
            ])
            return trans(image), label

    def __len__(self):
        # Orijinal FedAvg ağırlığını korumak için çoğaltılmış uzunluğu dönmek ZORUNDAYIZ
        return len(self.client_dataset)


class Indices2Dataset_unlabeled_fixmatch(Dataset):
    def __init__(self, dataset, dataset_name='CIFAR10'):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.indices = None

    def load(self, indices: list):
        self.indices = indices
        # Önce listeyi oluştur
        self.client_dataset = [self.dataset[i] for i in self.indices]
        
        # Sonra veri setine göre çoğalt
        if self.dataset_name == 'HAM10000':
            self.client_dataset *= 3 
        else:
            self.client_dataset *= 50 # Orijinal SAGE CIFAR çarpanı
            
        # Uzunluğu FixMatch döngüsü için kaydet
        self.client_dataset_len = len(self.client_dataset)

    def fixmatch(self, image):
        if self.dataset_name == 'HAM10000':
            if torch.is_tensor(image): image = transforms.ToPILImage()(image)
            weak = transforms.Compose([
                transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                transforms.RandomCrop(size=224, padding=28, padding_mode='reflect'),
                transforms.ToTensor(), transforms.Normalize((0.763, 0.545, 0.570), (0.140, 0.152, 0.169))
            ])
            strong = transforms.Compose([
                transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                transforms.RandomCrop(size=224, padding=28, padding_mode='reflect'),
                RandAugmentMC(n=2, m=10),
                transforms.ToTensor(), transforms.Normalize((0.763, 0.545, 0.570), (0.140, 0.152, 0.169))
            ])
            return weak(image), strong(image)
        else:
            # STRICT ORIGINAL SAGE LOGIC FOR CIFAR
            weak = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect'),
                transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
            ])
            strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
                RandAugmentMC(n=2, m=10),
                transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
            ])
            return weak(image), strong(image)

    def __getitem__(self, idx):
        image, label = self.client_dataset[idx]
        image1, image2 = self.fixmatch(image)
        return image1, image2, label

    def __len__(self):
        # Orijinal SAGE: Client Drift olmasın diye iterasyonu unmultiplied boyutla sınırlar!
        return self.client_dataset_len
