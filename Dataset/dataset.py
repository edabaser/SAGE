import numpy as np
from torch.utils.data.dataset import Dataset
import copy
import math
import torch
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from .randaugment import RandAugmentMC
from Dataset.sample_dirichlet import clients_indices, clients_indices_unlabel
import time

def classify_label(dataset, num_classes: int):
    """
    HIZLI VERSIYON: Resimleri acmaz, sadece ImageFolder'ın 
    onceden hazırladıgı targets listesini kullanır.
    """
    list_label2indices = [[] for _ in range(num_classes)]
    
    # Eger dataset bir Subset ise targets'a ulasmak icin .dataset kullanmalıyız
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
        print(f'client {client} labeled number per class : {nums_data_labeled}')
        print(f'client {client} unlabeled number per class  : {nums_data_unlabeled}')
    return dict_per_client_labeled

def partition_train(list_label2indices: list, ipc):
    list_label2indices_labeled = []
    list_label2indices_unlabeled = []
    for indices in list_label2indices:
        idx_shuffle = np.random.permutation(indices)
        list_label2indices_labeled.append(idx_shuffle[:ipc])
        list_label2indices_unlabeled.append(idx_shuffle[ipc:])
    return list_label2indices_labeled, list_label2indices_unlabeled

def compute_clients_labeled_data_distribution(dataset, clients_indices_labeled, num_classes):
    dict_per_client_labeled = []
    nums_data_labeled = [0 for _ in range(num_classes)]
    for idx in clients_indices_labeled:
        label = dataset[idx][1]
        nums_data_labeled[label] += 1
    dict_per_client_labeled.append(nums_data_labeled)
    return dict_per_client_labeled

class Indices2Dataset_labeled(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = None
        self.mean = (0.763, 0.545, 0.570)
        self.std = (0.140, 0.152, 0.169)
        self.img_size = 224

    def load(self, indices: list):
        self.indices = indices
        self.client_dataset = [self.dataset[i] for i in indices]
        # self.client_dataset *= 2000 
        
        sample_img, _ = self.client_dataset[0]
        if hasattr(sample_img, 'size'): 
            w, h = sample_img.size
        elif torch.is_tensor(sample_img): 
            w, h = sample_img.shape[-1], sample_img.shape[-2]
        else: 
            w, h = 224, 224

        self.img_size = w
        if self.img_size <= 32: # CIFAR-10
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std = (0.2471, 0.2435, 0.2616)  #(0.2023, 0.1994, 0.2010)
        else: # HAM10000
            self.mean = (0.763, 0.545, 0.570)
            self.std = (0.140, 0.152, 0.169)

    def __getitem__(self, idx):
        # DINAMIK TRANSFORM LISTESI
        transform_list = [transforms.RandomHorizontalFlip()]
        
        # Sadece medikal veri (boyut > 32) ise tepe taklak etmeye (Vertical Flip) izin ver
        if self.img_size > 32:
            transform_list.append(transforms.RandomVerticalFlip())
            
        transform_list.extend([
            transforms.RandomCrop(size=self.img_size,
                                  padding=int(self.img_size * 0.125),
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        self.label_trans = transforms.Compose(transform_list)

        image, label = self.client_dataset[idx]
        
        if torch.is_tensor(image):
            image = transforms.ToPILImage()(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image = self.label_trans(image)
        return image, label

    def __len__(self):
        return len(self.client_dataset)

class Indices2Dataset_unlabeled_fixmatch(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = None
        self.mean = (0.763, 0.545, 0.570)
        self.std = (0.140, 0.152, 0.169)
        self.img_size = 224

    def load(self, indices: list):
        self.indices = indices
        self.client_dataset = [self.dataset[i] for i in self.indices]
        self.client_dataset_len = len(self.client_dataset)
        self.client_dataset *= 50 
        
        sample_img, _ = self.client_dataset[0]
        if hasattr(sample_img, 'size'):
            w, h = sample_img.size
        else:
            w, h = 224, 224
            
        self.img_size = w
        if self.img_size <= 32:
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std = (0.2471, 0.2435, 0.2616) #(0.2023, 0.1994, 0.2010)
        else:
            self.mean = (0.763, 0.545, 0.570)
            self.std = (0.140, 0.152, 0.169)

    # def fixmatch(self, image):
    #     # DINAMIK TRANSFORM LISTESI (WEAK & STRONG ICIN)
    #     base_transforms = [transforms.RandomHorizontalFlip()]
        
    #     # Sadece HAM10000 ise dikey cevirmeyi ekle
    #     if self.img_size > 32:
    #         base_transforms.append(transforms.RandomVerticalFlip())
            
    #     base_transforms.append(
    #         transforms.RandomCrop(size=self.img_size,
    #                               padding=int(self.img_size * 0.125),
    #                               padding_mode='reflect')
    #     )

    #     self.weak = transforms.Compose(base_transforms)

    #     # Strong transform, zayif transformun ustune RandAugment ekler
    #     strong_transforms = base_transforms.copy()
    #     strong_transforms.append(RandAugmentMC(n=2, m=10))
    #     self.strong = transforms.Compose(strong_transforms)

    #     self.normalize = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=self.mean, std=self.std)
    #     ])

    #     if torch.is_tensor(image):
    #         image = transforms.ToPILImage()(image)
        
    #     weak = self.weak(image)
    #     strong = self.strong(image)
    #     return self.normalize(weak), self.normalize(strong)

    # def __getitem__(self, idx):
    #     image, label = self.client_dataset[idx]
    #     image1, image2 = self.fixmatch(image)
    #     return image1, image2, label

    # def __len__(self):
    #     # return len(self.client_dataset)
    #     return self.client_dataset_len


    def fixmatch(self, image):
        if torch.is_tensor(image):
            image = transforms.ToPILImage()(image)
    
        if self.img_size <= 32:  # CIFAR-10
            self.weak = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=self.img_size,
                                      padding=int(self.img_size * 0.125),
                                      padding_mode='reflect'),
            ])
            self.strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=self.img_size,
                                      padding=int(self.img_size * 0.125),
                                      padding_mode='reflect'),
                RandAugmentMC(n=2, m=10),
            ])
        else:  # HAM10000
            self.weak = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomCrop(size=self.img_size,
                                      padding=int(self.img_size * 0.125),
                                      padding_mode='reflect'),
            ])
            self.strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomCrop(size=self.img_size,
                                      padding=int(self.img_size * 0.125),
                                      padding_mode='reflect'),
                RandAugmentMC(n=2, m=10),
            ])
    
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
        weak = self.weak(image)
        strong = self.strong(image)
        return self.normalize(weak), self.normalize(strong)
    
    def __getitem__(self, idx):
        image, label = self.client_dataset[idx]
        image1, image2 = self.fixmatch(image)
        return image1, image2, label
    
    def __len__(self):
        return self.client_dataset_len 
    
def sampling_unlabeled_data_non_iid(args, list_label2indices_unlabeled, num_unlabeled_client, alpha, seed=0):
    list_choose_unlabeled = []
    list_unlabeled_part1 = []
    list_unlabeled_part2 = []
    random_state = np.random.RandomState(0)
    
    # Adjusted for HAM10000 class count (7 classes)
    class_sampling = [2000] * args.num_classes
    
    for class_idx, list_index in enumerate(list_label2indices_unlabeled):
        if class_idx < len(class_sampling):
            new_data = set(random_state.choice(list_index, min(len(list_index), class_sampling[class_idx]), replace=False))
            list_new_data = list(new_data)
            list_unlabeled_part1.append(list_new_data)
            list_index = list(set(list_index) - new_data)
            list_unlabeled_part2.append(list_index)

    list_client_part1 = clients_indices_unlabel(list_label2indices=list_unlabeled_part1,
                                                num_classes=args.num_classes, num_clients=10,
                                                non_iid_alpha=alpha, seed=1000)  
    list_client_part2 = clients_indices_unlabel(list_label2indices=list_unlabeled_part2, 
                                                num_classes=args.num_classes, num_clients=10,
                                                non_iid_alpha=alpha, seed=1000) 

    list_choose_unlabeled = list_client_part1 + list_client_part2
    return list_choose_unlabeled
