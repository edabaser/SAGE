import numpy as np
from torch.utils.data.dataset import Dataset
import torch
from torchvision import transforms
from .randaugment import RandAugmentMC
from collections import Counter


def classify_label(dataset, num_classes: int):
    list_label2indices = [[] for _ in range(num_classes)]
    targets = getattr(dataset, 'targets', None)
    if targets is None and hasattr(dataset, 'base_dataset'):
        targets = dataset.base_dataset.targets
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
    """
    Her sınıftan ipc adet etiketli, kalanı etiketlenmemiş olarak ayırır.
    ipc: images per class (sınıf başına etiketli veri sayısı)
    """
    list_label2indices_labeled = []
    list_label2indices_unlabeled = []
    for indices in list_label2indices:
        idx_shuffle = np.random.permutation(indices)
        # İlk ipc adet etiketli, kalan etiketlenmemiş
        labeled = idx_shuffle[:ipc]
        unlabeled = idx_shuffle[ipc:]
        list_label2indices_labeled.append(labeled)
        list_label2indices_unlabeled.append(unlabeled)
    return list_label2indices_labeled, list_label2indices_unlabeled


# ──────────────────────────────────────────────
# HAM10000 Transform Sabitleri
# ──────────────────────────────────────────────
HAM_MEAN = (0.763, 0.545, 0.570)
HAM_STD  = (0.140, 0.152, 0.169)

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2471, 0.2435, 0.2616)


def _ham_weak_transform():
    """Zayıf augmentation: sadece geometrik, renk bozmaz."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=HAM_MEAN, std=HAM_STD),
    ])


def _ham_strong_transform():
    """
    Güçlü augmentation: geometrik + hafif kontrast/parlaklık.
    Renk bozucu (Color, Solarize, Invert, Equalize) YÜKLÜ DEĞİL.
    Cutout boyutu 224px görüntüye uygun: 56px (%25).
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        RandAugmentMC(n=2, m=7),          # m=10 yerine 7: daha hafif augment
        transforms.ToTensor(),
        transforms.Normalize(mean=HAM_MEAN, std=HAM_STD),
    ])


def _ham_labeled_transform():
    """Labeled veri için eğitim transformu."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=HAM_MEAN, std=HAM_STD),
    ])


def _cifar_weak_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ])


def _cifar_strong_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect'),
        RandAugmentMC(n=2, m=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ])


def _cifar_labeled_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ])


# ──────────────────────────────────────────────
# Labeled Dataset
# ──────────────────────────────────────────────
class Indices2Dataset_labeled(Dataset):
    """
    Labeled veri seti.

    DEĞİŞİKLİK: Artık client_dataset çarpılmıyor (önceden *= 3).
    Sebebi: Çarpma pseudo-label sayısını şişiriyor ve overfitting yapıyor.
    Bunun yerine WeightedRandomSampler ile nadir sınıflar daha çok örneklenir.
    """

    def __init__(self, dataset, dataset_name='CIFAR10'):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.indices = None
        self.sample_weights = None  # WeightedSampler için

    def load(self, indices: list):
        self.indices = list(indices)
        # Gerçek veri — çarpma YOK
        self.client_dataset = [(i, self.dataset.targets[i]) for i in self.indices]

        # Sınıf ağırlıkları hesapla (WeightedRandomSampler için)
        labels = [self.dataset.targets[i] for i in self.indices]
        class_counts = Counter(labels)
        # Nadir sınıfa yüksek ağırlık
        weights = []
        for lbl in labels:
            count = class_counts[lbl]
            weights.append(1.0 / (count + 1e-6))
        self.sample_weights = weights

    def __getitem__(self, idx):
        real_idx, label = self.client_dataset[idx]
        image, _ = self.dataset[real_idx]

        # PIL kontrolü
        if torch.is_tensor(image):
            image = transforms.ToPILImage()(image)

        if self.dataset_name == 'HAM10000':
            trans = _ham_labeled_transform()
        else:
            trans = _cifar_labeled_transform()

        return trans(image), label

    def __len__(self):
        return len(self.client_dataset)


# ──────────────────────────────────────────────
# Unlabeled Dataset (FixMatch)
# ──────────────────────────────────────────────
class Indices2Dataset_unlabeled_fixmatch(Dataset):
    """
    Unlabeled veri seti — zayıf + güçlü augmentation çifti döner.

    DEĞİŞİKLİK: Artık çarpılmıyor (önceden *= 3).
    local_iter hesabı gerçek veri boyutuna göre yapılıyor.
    Bu sayede pseudo-label sayısı gerçekçi kalıyor.
    """

    def __init__(self, dataset, dataset_name='CIFAR10'):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.indices = None

    def load(self, indices: list):
        self.indices = list(indices)
        # Gerçek veri — çarpma YOK
        self.client_dataset = [(i, self.dataset.targets[i]) for i in self.indices]
        self.client_dataset_len = len(self.client_dataset)

    def fixmatch(self, image):
        if torch.is_tensor(image):
            image = transforms.ToPILImage()(image)

        if self.dataset_name == 'HAM10000':
            weak   = _ham_weak_transform()
            strong = _ham_strong_transform()
        else:
            weak   = _cifar_weak_transform()
            strong = _cifar_strong_transform()

        return weak(image), strong(image)

    def __getitem__(self, idx):
        real_idx, label = self.client_dataset[idx]
        image, _ = self.dataset[real_idx]
        image_w, image_s = self.fixmatch(image)
        return image_w, image_s, label

    def __len__(self):
        # Orijinal SAGE mantığı: gerçek boyut (çarpılmış değil)
        return self.client_dataset_len
