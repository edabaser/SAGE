from torchvision import datasets
from torchvision.transforms import transforms
from options import args_parser
from Dataset.dataset import classify_label, show_clients_data_distribution, Indices2Dataset_labeled, \
    Indices2Dataset_unlabeled_fixmatch, partition_train
from Dataset.sample_dirichlet import clients_indices, clients_indices_homo
import numpy as np
import pandas as pd
from torch import max, eq, no_grad
from Dataset.CINIC10 import CINIC10
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from Model.resnet import ResNet
from tqdm import tqdm
import copy
import torch
import random
from torch.utils.data import DataLoader, RandomSampler
import logging
import os

# Google Drive Checkpoint Settings
# The SAVE_PATH here must match the one in the Colab cell.
# Example: /content/drive/MyDrive/Colab Notebooks/EE 401/SAGE-master.v1/Checkpoints
SAVE_PATH = '/content/drive/MyDrive/Colab Notebooks/EE 401/SAGE-master.v1/Checkpoints'


class Global(object):
    def __init__(self, args):
        self.model = ResNet(resnet_size=8, scaling=4,
                            save_activations=False, group_norm_num_groups=None,
                            freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)

        self.model.cuda(args.gpu_id)
        self.num_classes = args.num_classes

    def initialize_for_model_fusion(self, list_dicts_local_params: list, list_nums_local_data: list):
        # FedAvg implementation
        fedavg_global_params = copy.deepcopy(list_dicts_local_params[0])
        for name_param in list_dicts_local_params[0]:
            list_values_param = []
            for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
                list_values_param.append(dict_local_params[name_param] * num_local_data)
            value_global_param = sum(list_values_param) / sum(list_nums_local_data)
            fedavg_global_params[name_param] = value_global_param
        return fedavg_global_params

    def fedavg_eval(self, fedavg_params, data_test, batch_size_test):
        self.model.load_state_dict(fedavg_params)
        self.model.eval()
        with no_grad():
            test_loader = DataLoader(data_test, batch_size_test)
            num_corrects = 0
            for data_batch in test_loader:
                images, labels = data_batch
                images, labels = images.cuda(args.gpu_id), labels.cuda(args.gpu_id)
                _, outputs = self.model(images)
                _, predicts = max(outputs, -1)
                num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
            accuracy = num_corrects / len(data_test)
        return accuracy

    def download_params(self):
        return self.model.state_dict()


class Local(object):
    def __init__(self, args):

        self.local_model = ResNet(resnet_size=8, scaling=4,
                                  save_activations=False, group_norm_num_groups=None,
                                  freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)


        self.local_G = ResNet(resnet_size=8, scaling=4,
                              save_activations=False, group_norm_num_groups=None,
                              freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)

        self.local_model.cuda(args.gpu_id)
        self.local_G.cuda(args.gpu_id)

        self.criterion = CrossEntropyLoss().cuda(args.gpu_id)
        # Optimizer is initialized here, but recreated in a common FixMatch pattern for each round.
        self.optimizer = SGD(self.local_model.parameters(), lr=args.lr_local_training, momentum=0.9, weight_decay=1e-4)

    def fixmatch_train(self, args, data_client_labeled, data_client_unlabeled, global_params, r):

        self.labeled_trainloader = DataLoader(
            dataset=data_client_labeled,
            sampler=RandomSampler(data_client_labeled),
            batch_size=args.batch_size_local_labeled_fixmatch,
            drop_last=True,
            num_workers = 2,
            pin_memory=True
        )

        self.unlabeled_trainloader = DataLoader(
            dataset=data_client_unlabeled,
            sampler=RandomSampler(data_client_unlabeled),
            batch_size=args.batch_size_local_labeled_fixmatch * args.mu,
            drop_last=True,
            num_workers=2,
            pin_memory=True
        )

        # Load the global model parameters for local training start
        self.local_model.load_state_dict(global_params)
        self.local_model.train()

        self.local_G.load_state_dict(global_params)
        self.local_G.eval()


        # default: E = 5
        for local_epoch in range(args.local_epochs):

            labeled_iter = iter(self.labeled_trainloader)
            unlabeled_iter = iter(self.unlabeled_trainloader)

            local_iter = int(len(data_client_unlabeled) / args.batch_size_local_labeled_fixmatch)

            for epoch in range(local_iter):

                try:
                    inputs_x, targets_x = labeled_iter.__next__()
                except:
                    labeled_iter = iter(self.labeled_trainloader)
                    inputs_x, targets_x = labeled_iter.__next__()

                try:
                    inputs_u_w, inputs_u_s, targets_u_groundtruth = unlabeled_iter.__next__()
                except:
                    unlabeled_iter = iter(self.unlabeled_trainloader)
                    inputs_u_w, inputs_u_s, targets_u_groundtruth = unlabeled_iter.__next__()

                inputs_x, inputs_u_w, inputs_u_s = inputs_x.cuda(args.gpu_id), inputs_u_w.cuda(args.gpu_id), inputs_u_s.cuda(args.gpu_id)

                batch_size = inputs_x.shape[0]
                inputs = self.interleave(
                    torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1).cuda(args.gpu_id)
                targets_x = targets_x.cuda(args.gpu_id)

                # local logits
                _, logits = self.local_model(inputs)
                logits = self.de_interleave(logits, 2 * args.mu + 1)
                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)

                del logits

                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')


                # global pseudo-labeling
                _, logits_u_w_global = self.local_G(inputs_u_w.cuda(args.gpu_id))
                pseudo_label_global = torch.softmax(logits_u_w_global.detach() / args.T, dim=-1)
                max_probs_global, targets_u_global = torch.max(pseudo_label_global, dim=-1)

                # local pseudo-labeling
                pseudo_label_local = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
                max_probs_local, targets_u_local = torch.max(pseudo_label_local, dim=-1)

                targets_u_local_one_hot = F.one_hot(targets_u_local, args.num_classes).float()
                targets_u_global_one_hot = F.one_hot(targets_u_global, args.num_classes).float()


                mask_local = max_probs_local.ge(args.threshold).float()
                mask_global = max_probs_global.ge(args.threshold).float()

                # delta_p
                delta_c = torch.abs(max_probs_local - max_probs_global) + 1e-6
                delta_c = torch.clamp(delta_c, min=1e-6, max=1.0)

                # Confidence-Driven Soft Correction
                # Set sensitivity parameter kappa
                # Ensure lambda_dynamic = 0.5 when delta_p = 0.05
                # kappa = log(2) / 0.05 based on the formula lambda_dynamic = exp(-kappa * delta_p)
                kappa = torch.log(torch.tensor(2.0)) / 0.05  # calculate kappa
                lambda_dynamic = torch.exp(-kappa * delta_c)
                lambda_dynamic = torch.clamp(lambda_dynamic, min=1e-6, max=1.0)

                final_targets_u = torch.where(
                    mask_local.unsqueeze(1).bool(),
                    lambda_dynamic.unsqueeze(1) * targets_u_local_one_hot + (1-lambda_dynamic).unsqueeze(1) * targets_u_global_one_hot,
                    targets_u_global_one_hot
                )

                mask_valid = torch.max(mask_local, mask_global)

                logits_u_s_probs = torch.softmax(logits_u_s, dim=-1)

                logits_u_s_probs = torch.softmax(logits_u_s, dim=-1) + 1e-10
                final_targets_u = final_targets_u + 1e-10
                Lu = (F.kl_div(logits_u_s_probs.log(), final_targets_u, reduction='none').sum(-1) * mask_valid).mean()


                loss = Lx + args.lambda_u * Lu

                # logging
                logging.info(
                    f'Round {r}, Local Epoch {local_epoch}, Batch {epoch}: Lx = {Lx.item():.4f}, Lu = {Lu.item():.4f}, lambda_dynamic = {lambda_dynamic[0].item():.4f}')


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return copy.deepcopy(self.local_model.state_dict())

    def interleave(self, x, size):
        s = list(x.shape)
        return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

    def de_interleave(self, x, size):
        s = list(x.shape)
        return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

# --- CHECKPOINT AND LOADING FUNCTIONS ---

def save_checkpoint(filename, model_state_dict, optimizer_state_dict, round_num, fedavg_acc):
    """Saves the training state to a file."""
    torch.save({
        'round': round_num,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'fedavg_acc': fedavg_acc,
    }, filename)
    print(f"\n🚀 Checkpoint saved. Round: {round_num}")

def load_checkpoint(filename, model, optimizer):
    """Loads the saved training state and returns the round number and accuracy history."""
    if os.path.exists(filename):
        print(f"\n✅ Checkpoint found and loading: {filename}")
        try:
            checkpoint = torch.load(filename, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            model.model.load_state_dict(checkpoint['model_state_dict'])

            # The optimizer state should normally be loaded here, but since Local.optimizer is recreated locally in every round
            # (a common practice in FixMatch), we only load the Global model and history.
            # The Local model/optimizer is updated just before starting local training.
            # That's why we only load the Global model state.
            start_round = checkpoint['round'] + 1
            fedavg_acc = checkpoint['fedavg_acc']
            print(f"✅ Successfully loaded. Continuing training from Round {start_round}.")
            return start_round, fedavg_acc
        except Exception as e:
            print(f"❌ An error occurred while loading the checkpoint ({e}). Restarting from the beginning.")
            return 1, []
    print("\n⚠️ Checkpoint not found. Starting training from Round 1.")
    return 1, []

# --- MAIN TRAINING LOOP (fixmatch) ---

def fixmatch(alpha):

    args = args_parser()

    # Setting up Log and Checkpoint directories
    log_dir = f'./results/{args.dataset}/logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'SAGE α={alpha}.log'.format(alpha = alpha))

    # Checkpoint directory: DATASET_aALPHA
    checkpoint_dir = os.path.join(SAVE_PATH, f'{args.dataset}_a{alpha}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pt')


    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=log_file
                        )

    if args.dataset == 'CIFAR10':
        args.num_classes = 10
        args.num_labeled = 500
        args.num_rounds = 300
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        data_local_training = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=None)
        data_global_test = datasets.CIFAR10(args.path_cifar10, train=False, transform=transform_test)

    elif args.dataset == 'CIFAR100':
        # ... (dataset configuration remains the same)
        args.num_classes = 100
        args.num_labeled = 50
        args.num_rounds = 500
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        data_local_training = datasets.CIFAR100(args.path_cifar100, train=True, download=True, transform=None)
        data_global_test = datasets.CIFAR100(args.path_cifar100, train=False, transform=transform_test)


    elif args.dataset == 'SVHN':
        # ... (dataset configuration remains the same)
        args.num_classes = 10
        args.num_labeled = 460
        args.num_rounds = 150
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        data_local_training = datasets.SVHN(args.path_svhn, split='train', download=True, transform=None)
        data_global_test = datasets.SVHN(args.path_svhn, split='test', transform=transform_test, download=True)

    elif args.dataset == 'CINIC10':
        # ... (dataset configuration remains the same)
        args.num_classes = 10
        args.num_labeled = 900
        args.num_rounds = 400
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4789, 0.4723, 0.4305), (0.2421, 0.2383, 0.2587)),
        ])
        data_local_training = CINIC10(root=args.path_cinic10, split='train', transform=None)
        data_global_test = CINIC10(root=args.path_cinic10, split='test', transform=transform_test)

    else:
        print(
            f"Error: Unsupported dataset {args.dataset}. Please specify one of the following: CIFAR10, CIFAR100, CINIC10 or SVHN.")
        exit(1)

    print(
        'dataset:{dataset}\n'
        'num_classes:{num_classes}\n'
        'num_labeled:{num_labeled}\n'
        'non_iid:{alpha}\n'
        'mu:{mu}\n'
        'num_rounds:{num_rounds}\n'
        'batch_label:{batch_label}, batch_unlabel:{batch_unlabel}'.format(
            dataset=args.dataset,
            num_classes=args.num_classes,
