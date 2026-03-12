# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, RandomSampler
# from torchvision.datasets import ImageFolder
# import numpy as np
# import pandas as pd
# import os
# import copy
# import random
# import logging
# from tqdm import tqdm
# from Model.resnet import ResNet
# from options import args_parser
# from Dataset.dataset import classify_label, show_clients_data_distribution, Indices2Dataset_labeled, Indices2Dataset_unlabeled_fixmatch, partition_train
# from Dataset.sample_dirichlet import clients_indices, clients_indices_homo
# from sklearn.metrics import recall_score, f1_score, confusion_matrix

# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True


# def save_checkpoint(round_num, model_state, fedavg_acc_history, checkpoint_dir, filename='checkpoint.pt'):
#     os.makedirs(checkpoint_dir, exist_ok=True)
#     state = {
#         'round': round_num,
#         'model_state_dict': model_state,
#         'fedavg_acc': fedavg_acc_history,
#     }
#     filepath = os.path.join(checkpoint_dir, filename)
#     torch.save(state, filepath)
#     print(f"\n[SAGE] Checkpoint saved at Round {round_num} to {filepath}")

# def load_checkpoint(model, checkpoint_dir, filename='checkpoint.pt'):
#     filepath = os.path.join(checkpoint_dir, filename)
#     if os.path.exists(filepath):
#         print(f"\n[SAGE] Checkpoint found at {filepath}. Loading...")
#         try:
#             checkpoint = torch.load(filepath, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
#             model.load_state_dict(checkpoint['model_state_dict'])
#             start_round = checkpoint['round'] + 1
#             fedavg_acc = checkpoint['fedavg_acc']
#             print(f"[SAGE] Resuming from Round {start_round}. Last Acc: {fedavg_acc[-1] if fedavg_acc else 0:.4f}")
#             return start_round, fedavg_acc
#         except Exception as e:
#             print(f"[SAGE] Error loading checkpoint: {e}. Starting from scratch.")
#             return 1, []
#     else:
#         print("[SAGE] No checkpoint found. Starting from Round 1.")
#         return 1, []

# def compute_cssv(args, local_models_params, initial_global_params):
#     num_clients = len(local_models_params)
#     if num_clients == 0:
#         return np.array([])
    
#     weight_layer = 'classifier.weight'
#     bias_layer = 'classifier.bias'
    
#     client_updates = []
#     for local_params in local_models_params:
#         update = {}
#         for name in local_params:
#             if name in initial_global_params:
#                 update[name] = local_params[name] - initial_global_params[name].to(local_params[name].device)
#         client_updates.append(update)

#     shapley_values = np.zeros(num_clients)
#     num_samples = args.shapley_samples if hasattr(args, 'shapley_samples') else 10 
    
#     for _ in range(num_samples):
#         permutation = np.random.permutation(num_clients)
#         for i, client_idx in enumerate(permutation):
#             coalition_indices = permutation[:i]
#             coalition_plus_indices = permutation[:i+1]
            
#             curr_update_w = torch.cat([
#                 client_updates[client_idx][weight_layer].view(-1),
#                 client_updates[client_idx][bias_layer].view(-1)
#             ])
#             curr_update_w_norm = F.normalize(curr_update_w.unsqueeze(0), p=2)

#             sim_s = 0.0
#             if len(coalition_indices) > 0:
#                 temp_w_s = torch.zeros_like(client_updates[0][weight_layer])
#                 temp_b_s = torch.zeros_like(client_updates[0][bias_layer])
#                 for c_idx in coalition_indices:
#                     temp_w_s += client_updates[c_idx][weight_layer]
#                     temp_b_s += client_updates[c_idx][bias_layer]
#                 temp_w_s /= len(coalition_indices)
#                 temp_b_s /= len(coalition_indices)
#                 agg_update_s_w = torch.cat([temp_w_s.view(-1), temp_b_s.view(-1)])
#                 agg_update_s_w_norm = F.normalize(agg_update_s_w.unsqueeze(0), p=2)
#                 sim_s = F.cosine_similarity(curr_update_w_norm, agg_update_s_w_norm).item()

#             temp_w_s_plus_i = torch.zeros_like(client_updates[0][weight_layer])
#             temp_b_s_plus_i = torch.zeros_like(client_updates[0][bias_layer])
#             for c_idx in coalition_plus_indices:
#                 temp_w_s_plus_i += client_updates[c_idx][weight_layer]
#                 temp_b_s_plus_i += client_updates[c_idx][bias_layer]
#             temp_w_s_plus_i /= len(coalition_plus_indices)
#             temp_b_s_plus_i /= len(coalition_plus_indices)
#             agg_update_s_plus_i_w = torch.cat([temp_w_s_plus_i.view(-1), temp_b_s_plus_i.view(-1)])
#             agg_update_s_plus_i_w_norm = F.normalize(agg_update_s_plus_i_w.unsqueeze(0), p=2)
#             sim_s_plus_i = F.cosine_similarity(curr_update_w_norm, agg_update_s_plus_i_w_norm).item()
            
#             marginal_contribution = sim_s_plus_i - sim_s
#             shapley_values[client_idx] += marginal_contribution

#     if num_samples > 0:
#         shapley_values /= num_samples

#     shapley_values = np.maximum(shapley_values, 0)
#     total_shapley = np.sum(shapley_values)
    
#     if total_shapley > 0:
#         normalized_weights = shapley_values / total_shapley
#     else:
#         normalized_weights = np.ones(num_clients) / num_clients
            
#     return normalized_weights

# class Global(object):
#     def __init__(self, args):
#         self.model = ResNet(resnet_size=8, scaling=4,
#                             save_activations=False, group_norm_num_groups=None,
#                             freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
#         self.model.cuda(args.gpu_id)
#         self.num_classes = args.num_classes

#     def initialize_for_model_fusion(self, args, list_dicts_local_params, list_nums_local_data, initial_global_params):
#         fused_params = copy.deepcopy(list_dicts_local_params[0])
#         if args.aggregation_method == 'ShapFed':
#             weights = compute_cssv(args, list_dicts_local_params, initial_global_params)
#         else:
#             total_data = sum(list_nums_local_data)
#             weights = [n / total_data for n in list_nums_local_data]
#         for name_param in list_dicts_local_params[0]:
#             list_values_param = []
#             for dict_local_params, weight in zip(list_dicts_local_params, weights):
#                 list_values_param.append(dict_local_params[name_param] * weight)
#             fused_params[name_param] = sum(list_values_param)
#         return fused_params

#     def fedavg_eval(self, fedavg_params, data_test, batch_size_test, args):
#         self.model.load_state_dict(fedavg_params)
#         self.model.eval()
#         all_labels = []
#         all_predicts = []
#         with torch.no_grad():
#             test_loader = DataLoader(data_test, batch_size_test)
#             num_corrects = 0
#             for data_batch in test_loader:
#                 images, labels = data_batch
#                 images, labels = images.cuda(args.gpu_id), labels.cuda(args.gpu_id)
#                 _, outputs = self.model(images)
#                 _, predicts = torch.max(outputs, -1)
#                 num_corrects += torch.sum(torch.eq(predicts.cpu(), labels.cpu())).item()
#                 all_labels.extend(labels.cpu().numpy())
#                 all_predicts.extend(predicts.cpu().numpy())
#             accuracy = num_corrects / len(data_test)
#             acsa = recall_score(all_labels, all_predicts, average='macro', zero_division=0)
#             macro_f1 = f1_score(all_labels, all_predicts, average='macro', zero_division=0)
#         return accuracy, acsa, macro_f1

#     def download_params(self):
#         return self.model.state_dict()

# class Local(object):
#     def __init__(self, args):
#         self.local_model = ResNet(resnet_size=8, scaling=4,
#                                   save_activations=False, group_norm_num_groups=None,
#                                   freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
#         self.local_G = ResNet(resnet_size=8, scaling=4,
#                               save_activations=False, group_norm_num_groups=None,
#                               freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
#         self.local_model.cuda(args.gpu_id)
#         self.local_G.cuda(args.gpu_id)
#         self.criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu_id)
#         self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=args.lr_local_training, momentum=0.9, weight_decay=1e-4)

#     def fixmatch_train(self, args, data_client_labeled, data_client_unlabeled, global_params, r):
#         self.labeled_trainloader = DataLoader(
#             dataset=data_client_labeled, sampler=RandomSampler(data_client_labeled),
#             batch_size=args.batch_size_local_labeled_fixmatch, drop_last=True, num_workers=2, pin_memory=True
#         )
#         self.unlabeled_trainloader = DataLoader(
#             dataset=data_client_unlabeled, sampler=RandomSampler(data_client_unlabeled),
#             batch_size=args.batch_size_local_labeled_fixmatch * args.mu, drop_last=True, num_workers=2, pin_memory=True
#         )
#         self.local_model.load_state_dict(global_params)
#         self.local_model.train()
#         self.local_G.load_state_dict(global_params)
#         self.local_G.eval()
#         for local_epoch in range(args.local_epochs):
#             labeled_iter = iter(self.labeled_trainloader)
#             unlabeled_iter = iter(self.unlabeled_trainloader)
#             local_iter = int(len(data_client_unlabeled) / args.batch_size_local_labeled_fixmatch)
#             for epoch in range(local_iter):
#                 try: inputs_x, targets_x = labeled_iter.__next__()
#                 except: labeled_iter = iter(self.labeled_trainloader); inputs_x, targets_x = labeled_iter.__next__()
#                 try: inputs_u_w, inputs_u_s, targets_u_groundtruth = unlabeled_iter.__next__()
#                 except: unlabeled_iter = iter(self.unlabeled_trainloader); inputs_u_w, inputs_u_s, targets_u_groundtruth = unlabeled_iter.__next__()
#                 inputs_x, inputs_u_w, inputs_u_s = inputs_x.cuda(args.gpu_id), inputs_u_w.cuda(args.gpu_id), inputs_u_s.cuda(args.gpu_id)
#                 batch_size = inputs_x.shape[0]
#                 inputs = self.interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1).cuda(args.gpu_id)
#                 targets_x = targets_x.cuda(args.gpu_id)
#                 _, logits = self.local_model(inputs)
#                 logits = self.de_interleave(logits, 2 * args.mu + 1)
#                 logits_x = logits[:batch_size]
#                 logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
#                 del logits
#                 Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
#                 _, logits_u_w_global = self.local_G(inputs_u_w.cuda(args.gpu_id))
#                 pseudo_label_global = torch.softmax(logits_u_w_global.detach() / args.T, dim=-1)
#                 max_probs_global, targets_u_global = torch.max(pseudo_label_global, dim=-1)
#                 pseudo_label_local = torch.softmax(logits_u_w.detach() / args.T, dim=-1)Max_probs_local, targets_u_local = torch.max(pseudo_label_local, dim=-1)
#                 targets_u_local_one_hot = F.one_hot(targets_u_local, args.num_classes).float()
#                 targets_u_global_one_hot = F.one_hot(targets_u_global, args.num_classes).float()
#                 mask_local = max_probs_local.ge(args.threshold).float()
#                 mask_global = max_probs_global.ge(args.threshold).float()
#                 delta_c = torch.clamp(torch.abs(max_probs_local - max_probs_global) + 1e-6, min=1e-6, max=1.0)
#                 kappa = torch.log(torch.tensor(2.0)) / 0.05
#                 lambda_dynamic = torch.clamp(torch.exp(-kappa * delta_c), min=1e-6, max=1.0)
#                 final_targets_u = torch.where(
#                     mask_local.unsqueeze(1).bool(),
#                     lambda_dynamic.unsqueeze(1) * targets_u_local_one_hot + (1 - lambda_dynamic).unsqueeze(1) * targets_u_global_one_hot,
#                     targets_u_global_one_hot
#                 )
#                 mask_valid = torch.max(mask_local, mask_global)
#                 logits_u_s_probs = torch.softmax(logits_u_s, dim=-1) + 1e-10
#                 final_targets_u = final_targets_u + 1e-10
#                 Lu = (F.kl_div(logits_u_s_probs.log(), final_targets_u, reduction='none').sum(-1) * mask_valid).mean()
#                 loss = Lx + args.lambda_u * Lu
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#         return copy.deepcopy(self.local_model.state_dict())

#     def interleave(self, x, size):
#         s = list(x.shape)
#         return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

#     def de_interleave(self, x, size):
#         s = list(x.shape)
#         return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

# def main_loop(alpha):
#     args = args_parser()
#     checkpoint_dir = os.path.join(args.checkpoint_dir, f'{args.dataset}_a{alpha}_{args.aggregation_method}')
#     os.makedirs(checkpoint_dir, exist_ok=True)
#     print(f"Checkpoints will be saved to: {checkpoint_dir}")
#     log_dir = f'./results/{args.dataset}/logs'
#     os.makedirs(log_dir, exist_ok=True)
#     log_file = os.path.join(log_dir, f'SAGE_{args.aggregation_method}_alpha={alpha}.log')
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file)
#     if args.dataset == 'CIFAR10':
#         args.num_classes = 10
#         args.num_labeled = 500 
#         args.num_rounds = 600 
#         transform_test = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#         ])
#         data_local_training = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=None)
#         data_global_test = datasets.CIFAR10(args.path_cifar10, train=False, transform=transform_test)
#     elif args.dataset == 'CIFAR100':
#         args.num_classes = 100
#         args.num_labeled = 50
#         args.num_rounds = 500
#         transform_test = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
#         ])
#         data_local_training = datasets.CIFAR100(args.path_cifar100, train=True, download=True, transform=None)
#         data_global_test = datasets.CIFAR100(args.path_cifar100, train=False, transform=transform_test)
#     elif args.dataset == 'SVHN':
#         args.num_classes = 10
#         args.num_labeled = 460
#         args.num_rounds = 300
#         transform_test = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
#         ])
#         data_local_training = datasets.SVHN(args.path_svhn, split='train', download=True, transform=None)
#         data_global_test = datasets.SVHN(args.path_svhn, split='test', transform=transform_test, download=True)
#     elif args.dataset == 'CINIC10':
#         args.num_classes = 10
#         args.num_labeled = 900
#         args.num_rounds = 400
#         transform_test = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.4789, 0.4723, 0.4305), (0.2421, 0.2383, 0.2587)),
#         ])
#         data_local_training = CINIC10(root=args.path_cinic10, split='train', transform=None)
#         data_global_test = CINIC10(root=args.path_cinic10, split='test', transform=transform_test)
#     elif args.dataset == 'HAM10000':
#         args.num_classes = 7
#         args.num_labeled = 100 
#         args.num_rounds = 400   
#         ham_mean = [0.763, 0.545, 0.570]
#         ham_std = [0.140, 0.152, 0.169]
#         transform_test = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=ham_mean, std=ham_std)
#         ])
#         data_local_training = ImageFolder(root=args.path_ham10000, transform=None)
#         data_global_test = ImageFolder(root=args.path_ham10000, transform=transform_test)
#     else:
#         exit(1)
#     random_state = np.random.RandomState(args.seed)
#     list_label2indices = classify_label(data_local_training, args.num_classes)
#     list_label2indices_labeled, list_label2indices_unlabeled = partition_train(list_label2indices, args.num_labeled)
#     if alpha == 0:
#         list_client2indices_labeled = clients_indices_homo(list_label2indices=list_label2indices_labeled, num_classes=args.num_classes, num_clients=args.num_clients)
#         list_client2indices_unlabeled = clients_indices_homo(list_label2indices=list_label2indices_unlabeled, num_classes=args.num_classes, num_clients=args.num_clients)
#     else:
#         list_client2indices_labeled = clients_indices(list_label2indices=list_label2indices_labeled, num_classes=args.num_classes, num_clients=args.num_clients, non_iid_alpha=alpha, seed=0)
#         list_client2indices_unlabeled = clients_indices(list_label2indices=list_label2indices_unlabeled, num_classes=args.num_classes, num_clients=args.num_clients, non_iid_alpha=alpha, seed=0)
#     for client in range(args.num_clients):
#         list_client2indices_unlabeled[client].extend(list_client2indices_labeled[client])
#     global_model = Global(args)
#     local_model = Local(args)
#     metrics_history = {
#     'acc': [],
#     'acsa': [],
#     'f1': []
#     }
#     start_round, fedavg_acc = load_checkpoint(global_model.model, checkpoint_dir)
#     total_clients = list(range(args.num_clients))
#     indices2data_labeled = Indices2Dataset_labeled(data_local_training)
#     indices2data_unlabeled = Indices2Dataset_unlabeled_fixmatch(data_local_training)
#     for r in tqdm(range(start_round, args.num_rounds + 1), desc='Server'):
#         dict_global_params = global_model.download_params()
#         online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
#         list_dicts_local_params = []
#         list_nums_local_data = []
#         for client in online_clients:
#             indices2data_labeled.load(list_client2indices_labeled[client])
#             data_client_labeled = indices2data_labeled
#             indices2data_unlabeled.load(list_client2indices_unlabeled[client])
#             data_client_unlabeled = indices2data_unlabeled
#             list_nums_local_data.append(len(data_client_labeled) + len(data_client_unlabeled))
#             local_params = local_model.fixmatch_train(args, data_client_labeled, data_client_unlabeled, copy.deepcopy(dict_global_params), r)
#             list_dicts_local_params.append(copy.deepcopy(local_params))
#         fedavg_params = global_model.initialize_for_model_fusion(args, list_dicts_local_params, list_nums_local_data, dict_global_params)
#         global_model.model.load_state_dict(fedavg_params)
#         global_acc = global_model.fedavg_eval(copy.deepcopy(fedavg_params), data_global_test, args.batch_size_test, args)
#         fedavg_acc.append(global_acc)
#         save_checkpoint(r, global_model.download_params(), fedavg_acc, checkpoint_dir)
#         result_dir = f'./results/{args.dataset}'
#         os.makedirs(result_dir, exist_ok=True)
#         result_file = f'{result_dir}/{args.aggregation_method}_alpha={alpha}.csv'
#         acc_df = pd.DataFrame({'acc': fedavg_acc}, index=list(range(1, len(fedavg_acc) + 1)))
#         acc_df.to_csv(result_file, encoding='utf8')

# if __name__ == '__main__':
#     torch.manual_seed(7)
#     torch.cuda.manual_seed(7)
#     np.random.seed(7)
#     random.seed(7)
#     torch.backends.cudnn.deterministic = True
#     args = args_parser()
#     main_loop(args.alpha)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets import ImageFolder
import numpy as np
import pandas as pd
import os
import copy
import random
import logging
from tqdm import tqdm
from Model.resnet import ResNet
from options import args_parser
from Dataset.dataset import classify_label, show_clients_data_distribution, Indices2Dataset_labeled, Indices2Dataset_unlabeled_fixmatch, partition_train
from Dataset.sample_dirichlet import clients_indices, clients_indices_homo
from sklearn.metrics import recall_score, f1_score

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ─────────────────────────────────────────────
#  CHECKPOINT  (acc + acsa + f1 hepsini saklar)
# ─────────────────────────────────────────────

def save_checkpoint(round_num, model_state, metrics_history, checkpoint_dir, filename='checkpoint.pt'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = {
        'round': round_num,
        'model_state_dict': model_state,
        'metrics_history': metrics_history,   # {'acc':[], 'acsa':[], 'f1':[]}
    }
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"[SAGE] Checkpoint saved → Round {round_num}  |  {filepath}")


def load_checkpoint(model, checkpoint_dir, filename='checkpoint.pt'):
    filepath = os.path.join(checkpoint_dir, filename)
    if os.path.exists(filepath):
        print(f"\n[SAGE] Checkpoint found: {filepath}  →  Loading...")
        try:
            checkpoint = torch.load(
                filepath,
                map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            start_round = checkpoint['round'] + 1

            # Eski checkpoint'lerde sadece fedavg_acc listesi varsa dönüştür
            if 'metrics_history' in checkpoint:
                metrics_history = checkpoint['metrics_history']
            elif 'fedavg_acc' in checkpoint:
                old = checkpoint['fedavg_acc']
                # Eski format: liste içi tuple (acc, acsa, f1) ya da sadece float
                if old and isinstance(old[0], (list, tuple)):
                    metrics_history = {
                        'acc':  [x[0] for x in old],
                        'acsa': [x[1] for x in old],
                        'f1':   [x[2] for x in old],
                    }
                else:
                    metrics_history = {'acc': old, 'acsa': [], 'f1': []}
            else:
                metrics_history = {'acc': [], 'acsa': [], 'f1': []}

            # Özet bilgi
            if metrics_history['acc']:
                best_acc  = max(metrics_history['acc'])
                best_acsa = max(metrics_history['acsa']) if metrics_history['acsa'] else 0.0
                best_f1   = max(metrics_history['f1'])   if metrics_history['f1']   else 0.0
                last_acc  = metrics_history['acc'][-1]
                print(f"[SAGE] Resuming from Round {start_round}")
                print(f"       Last  → Acc: {last_acc:.4f}")
                print(f"       Best  → Acc: {best_acc:.4f} | ACSA: {best_acsa:.4f} | F1: {best_f1:.4f}")
            return start_round, metrics_history

        except Exception as e:
            print(f"[SAGE] Checkpoint load error: {e}  →  Starting from scratch.")
            return 1, {'acc': [], 'acsa': [], 'f1': []}
    else:
        print("[SAGE] No checkpoint found. Starting from Round 1.")
        return 1, {'acc': [], 'acsa': [], 'f1': []}


# ─────────────────────────────────────────────
#  SHAPLEY (CSSV)
# ─────────────────────────────────────────────

def compute_cssv(args, local_models_params, initial_global_params):
    num_clients = len(local_models_params)
    if num_clients == 0:
        return np.array([])

    weight_layer = 'classifier.weight'
    bias_layer   = 'classifier.bias'

    client_updates = []
    for local_params in local_models_params:
        update = {}
        for name in local_params:
            if name in initial_global_params:
                update[name] = local_params[name] - initial_global_params[name].to(local_params[name].device)
        client_updates.append(update)

    shapley_values = np.zeros(num_clients)
    num_samples = args.shapley_samples if hasattr(args, 'shapley_samples') else 10

    for _ in range(num_samples):
        permutation = np.random.permutation(num_clients)
        for i, client_idx in enumerate(permutation):
            coalition_indices      = permutation[:i]
            coalition_plus_indices = permutation[:i+1]

            curr_update_w = torch.cat([
                client_updates[client_idx][weight_layer].view(-1),
                client_updates[client_idx][bias_layer].view(-1)
            ])
            curr_update_w_norm = F.normalize(curr_update_w.unsqueeze(0), p=2)

            sim_s = 0.0
            if len(coalition_indices) > 0:
                temp_w_s = torch.zeros_like(client_updates[0][weight_layer])
                temp_b_s = torch.zeros_like(client_updates[0][bias_layer])
                for c_idx in coalition_indices:
                    temp_w_s += client_updates[c_idx][weight_layer]
                    temp_b_s += client_updates[c_idx][bias_layer]
                temp_w_s /= len(coalition_indices)
                temp_b_s /= len(coalition_indices)
                agg_s = torch.cat([temp_w_s.view(-1), temp_b_s.view(-1)])
                sim_s = F.cosine_similarity(curr_update_w_norm, F.normalize(agg_s.unsqueeze(0), p=2)).item()

            temp_w_si = torch.zeros_like(client_updates[0][weight_layer])
            temp_b_si = torch.zeros_like(client_updates[0][bias_layer])
            for c_idx in coalition_plus_indices:
                temp_w_si += client_updates[c_idx][weight_layer]
                temp_b_si += client_updates[c_idx][bias_layer]
            temp_w_si /= len(coalition_plus_indices)
            temp_b_si /= len(coalition_plus_indices)
            agg_si = torch.cat([temp_w_si.view(-1), temp_b_si.view(-1)])
            sim_si = F.cosine_similarity(curr_update_w_norm, F.normalize(agg_si.unsqueeze(0), p=2)).item()

            shapley_values[client_idx] += (sim_si - sim_s)

    if num_samples > 0:
        shapley_values /= num_samples

    shapley_values = np.maximum(shapley_values, 0)
    total_shapley  = np.sum(shapley_values)
    if total_shapley > 0:
        return shapley_values / total_shapley
    else:
        return np.ones(num_clients) / num_clients


# ─────────────────────────────────────────────
#  GLOBAL MODEL
# ─────────────────────────────────────────────

class Global(object):
    def __init__(self, args):
        self.model = ResNet(resnet_size=8, scaling=4,
                            save_activations=False, group_norm_num_groups=None,
                            freeze_bn=False, freeze_bn_affine=False,
                            num_classes=args.num_classes)
        self.model.cuda(args.gpu_id)
        self.num_classes = args.num_classes

    def initialize_for_model_fusion(self, args, list_dicts_local_params, list_nums_local_data, initial_global_params):
        fused_params = copy.deepcopy(list_dicts_local_params[0])
        if args.aggregation_method == 'ShapFed':
            weights = compute_cssv(args, list_dicts_local_params, initial_global_params)
        else:
            total_data = sum(list_nums_local_data)
            weights = [n / total_data for n in list_nums_local_data]

        for name_param in list_dicts_local_params[0]:
            fused_params[name_param] = sum(
                d[name_param] * w for d, w in zip(list_dicts_local_params, weights)
            )
        return fused_params

    def fedavg_eval(self, fedavg_params, data_test, batch_size_test, args):
        self.model.load_state_dict(fedavg_params)
        self.model.eval()
        all_labels, all_predicts = [], []
        num_corrects = 0
        with torch.no_grad():
            for images, labels in DataLoader(data_test, batch_size_test):
                images, labels = images.cuda(args.gpu_id), labels.cuda(args.gpu_id)
                _, outputs = self.model(images)
                _, predicts = torch.max(outputs, -1)
                num_corrects += torch.sum(torch.eq(predicts.cpu(), labels.cpu())).item()
                all_labels.extend(labels.cpu().numpy())
                all_predicts.extend(predicts.cpu().numpy())
        accuracy   = num_corrects / len(data_test)
        acsa       = recall_score(all_labels, all_predicts, average='macro', zero_division=0)
        macro_f1   = f1_score(all_labels, all_predicts, average='macro', zero_division=0)
        return accuracy, acsa, macro_f1

    def download_params(self):
        return self.model.state_dict()


# ─────────────────────────────────────────────
#  LOCAL MODEL
# ─────────────────────────────────────────────

class Local(object):
    def __init__(self, args):
        self.local_model = ResNet(resnet_size=8, scaling=4,
                                  save_activations=False, group_norm_num_groups=None,
                                  freeze_bn=False, freeze_bn_affine=False,
                                  num_classes=args.num_classes)
        self.local_G = ResNet(resnet_size=8, scaling=4,
                              save_activations=False, group_norm_num_groups=None,
                              freeze_bn=False, freeze_bn_affine=False,
                              num_classes=args.num_classes)
        self.local_model.cuda(args.gpu_id)
        self.local_G.cuda(args.gpu_id)
        self.criterion  = torch.nn.CrossEntropyLoss().cuda(args.gpu_id)
        self.optimizer  = torch.optim.SGD(
            self.local_model.parameters(),
            lr=args.lr_local_training, momentum=0.9, weight_decay=1e-4
        )

    def fixmatch_train(self, args, data_client_labeled, data_client_unlabeled, global_params, r):
        self.labeled_trainloader = DataLoader(
            dataset=data_client_labeled,
            sampler=RandomSampler(data_client_labeled),
            batch_size=args.batch_size_local_labeled_fixmatch,
            drop_last=True, num_workers=2, pin_memory=True
        )
        self.unlabeled_trainloader = DataLoader(
            dataset=data_client_unlabeled,
            sampler=RandomSampler(data_client_unlabeled),
            batch_size=args.batch_size_local_labeled_fixmatch * args.mu,
            drop_last=True, num_workers=2, pin_memory=True
        )
        self.local_model.load_state_dict(global_params)
        self.local_model.train()
        self.local_G.load_state_dict(global_params)
        self.local_G.eval()

        for local_epoch in range(args.local_epochs):
            labeled_iter   = iter(self.labeled_trainloader)
            unlabeled_iter = iter(self.unlabeled_trainloader)
            local_iter = int(len(data_client_unlabeled) / args.batch_size_local_labeled_fixmatch)

            for epoch in range(local_iter):
                try:
                    inputs_x, targets_x = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(self.labeled_trainloader)
                    inputs_x, targets_x = next(labeled_iter)

                try:
                    inputs_u_w, inputs_u_s, targets_u_groundtruth = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(self.unlabeled_trainloader)
                    inputs_u_w, inputs_u_s, targets_u_groundtruth = next(unlabeled_iter)

                inputs_x   = inputs_x.cuda(args.gpu_id)
                inputs_u_w = inputs_u_w.cuda(args.gpu_id)
                inputs_u_s = inputs_u_s.cuda(args.gpu_id)
                targets_x  = targets_x.cuda(args.gpu_id)
                batch_size = inputs_x.shape[0]

                inputs = self.interleave(
                    torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1
                ).cuda(args.gpu_id)

                _, logits = self.local_model(inputs)
                logits    = self.de_interleave(logits, 2 * args.mu + 1)
                logits_x  = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                del logits

                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

                _, logits_u_w_global = self.local_G(inputs_u_w)
                pseudo_label_global  = torch.softmax(logits_u_w_global.detach() / args.T, dim=-1)
                max_probs_global, targets_u_global = torch.max(pseudo_label_global, dim=-1)

                # ── SYNTAX FIX: ayrı satıra alındı ──
                pseudo_label_local = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
                max_probs_local, targets_u_local = torch.max(pseudo_label_local, dim=-1)

                targets_u_local_one_hot  = F.one_hot(targets_u_local,  args.num_classes).float()
                targets_u_global_one_hot = F.one_hot(targets_u_global, args.num_classes).float()

                mask_local  = max_probs_local.ge(args.threshold).float()
                mask_global = max_probs_global.ge(args.threshold).float()

                delta_c        = torch.clamp(torch.abs(max_probs_local - max_probs_global) + 1e-6, min=1e-6, max=1.0)
                kappa          = torch.log(torch.tensor(2.0)) / 0.05
                lambda_dynamic = torch.clamp(torch.exp(-kappa * delta_c), min=1e-6, max=1.0)

                final_targets_u = torch.where(
                    mask_local.unsqueeze(1).bool(),
                    lambda_dynamic.unsqueeze(1) * targets_u_local_one_hot
                    + (1 - lambda_dynamic).unsqueeze(1) * targets_u_global_one_hot,
                    targets_u_global_one_hot
                )
                mask_valid = torch.max(mask_local, mask_global)

                logits_u_s_probs = torch.softmax(logits_u_s, dim=-1) + 1e-10
                final_targets_u  = final_targets_u + 1e-10

                Lu   = (F.kl_div(logits_u_s_probs.log(), final_targets_u, reduction='none').sum(-1) * mask_valid).mean()
                loss = Lx + args.lambda_u * Lu

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


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────

def main_loop(alpha):
    args = args_parser()

    checkpoint_dir = os.path.join(args.checkpoint_dir, f'{args.dataset}_a{alpha}_{args.aggregation_method}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints → {checkpoint_dir}")

    log_dir  = f'./results/{args.dataset}/logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'SAGE_{args.aggregation_method}_alpha={alpha}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='a'          # append — resume durumunda log kaybolmaz
    )

    # ── Dataset ──
    if args.dataset == 'CIFAR10':
        args.num_classes = 10
        args.num_labeled = 500
        args.num_rounds  = 600
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        data_local_training = datasets.CIFAR10(args.path_cifar10, train=True,  download=True, transform=None)
        data_global_test    = datasets.CIFAR10(args.path_cifar10, train=False, transform=transform_test)

    elif args.dataset == 'CIFAR100':
        args.num_classes = 100
        args.num_labeled = 50
        args.num_rounds  = 500
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        data_local_training = datasets.CIFAR100(args.path_cifar100, train=True,  download=True, transform=None)
        data_global_test    = datasets.CIFAR100(args.path_cifar100, train=False, transform=transform_test)

    elif args.dataset == 'SVHN':
        args.num_classes = 10
        args.num_labeled = 460
        args.num_rounds  = 300
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        data_local_training = datasets.SVHN(args.path_svhn, split='train', download=True, transform=None)
        data_global_test    = datasets.SVHN(args.path_svhn, split='test',  download=True, transform=transform_test)

    elif args.dataset == 'CINIC10':
        args.num_classes = 10
        args.num_labeled = 900
        args.num_rounds  = 400
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4789, 0.4723, 0.4305), (0.2421, 0.2383, 0.2587)),
        ])
        from Dataset.cinic10 import CINIC10
        data_local_training = CINIC10(root=args.path_cinic10, split='train', transform=None)
        data_global_test    = CINIC10(root=args.path_cinic10, split='test',  transform=transform_test)

    elif args.dataset == 'HAM10000':
        args.num_classes = 7
        args.num_labeled = 100
        args.num_rounds  = 400
        ham_mean = [0.763, 0.545, 0.570]
        ham_std  = [0.140, 0.152, 0.169]
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=ham_mean, std=ham_std),
        ])
        data_local_training = ImageFolder(root=args.path_ham10000, transform=None)
        data_global_test    = ImageFolder(root=args.path_ham10000, transform=transform_test)
    else:
        exit(1)

    # ── Veri dağılımı ──
    random_state = np.random.RandomState(args.seed)
    list_label2indices = classify_label(data_local_training, args.num_classes)
    list_label2indices_labeled, list_label2indices_unlabeled = partition_train(list_label2indices, args.num_labeled)

    if alpha == 0:
        list_client2indices_labeled   = clients_indices_homo(list_label2indices_labeled,   args.num_classes, args.num_clients)
        list_client2indices_unlabeled = clients_indices_homo(list_label2indices_unlabeled, args.num_classes, args.num_clients)
    else:
        list_client2indices_labeled   = clients_indices(list_label2indices_labeled,   args.num_classes, args.num_clients, alpha, seed=0)
        list_client2indices_unlabeled = clients_indices(list_label2indices_unlabeled, args.num_classes, args.num_clients, alpha, seed=0)

    for client in range(args.num_clients):
        list_client2indices_unlabeled[client].extend(list_client2indices_labeled[client])

    # ── Model + checkpoint ──
    global_model = Global(args)
    local_model  = Local(args)
    start_round, metrics_history = load_checkpoint(global_model.model, checkpoint_dir)

    total_clients         = list(range(args.num_clients))
    indices2data_labeled   = Indices2Dataset_labeled(data_local_training)
    indices2data_unlabeled = Indices2Dataset_unlabeled_fixmatch(data_local_training)

    # ── Training loop ──
    for r in tqdm(range(start_round, args.num_rounds + 1), desc='Server'):
        dict_global_params = global_model.download_params()
        online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)

        list_dicts_local_params = []
        list_nums_local_data    = []

        for client in online_clients:
            indices2data_labeled.load(list_client2indices_labeled[client])
            indices2data_unlabeled.load(list_client2indices_unlabeled[client])
            list_nums_local_data.append(
                len(indices2data_labeled) + len(indices2data_unlabeled)
            )
            local_params = local_model.fixmatch_train(
                args, indices2data_labeled, indices2data_unlabeled,
                copy.deepcopy(dict_global_params), r
            )
            list_dicts_local_params.append(copy.deepcopy(local_params))

        fedavg_params = global_model.initialize_for_model_fusion(
            args, list_dicts_local_params, list_nums_local_data, dict_global_params
        )
        global_model.model.load_state_dict(fedavg_params)

        # ── Evaluation ──
        acc, acsa, macro_f1 = global_model.fedavg_eval(
            copy.deepcopy(fedavg_params), data_global_test, args.batch_size_test, args
        )
        metrics_history['acc'].append(acc)
        metrics_history['acsa'].append(acsa)
        metrics_history['f1'].append(macro_f1)

        best_acc  = max(metrics_history['acc'])
        best_acsa = max(metrics_history['acsa'])
        best_f1   = max(metrics_history['f1'])

        # ── Round sonu konsol çıktısı ──
        print(
            f"\n[Round {r:>4}/{args.num_rounds}] "
            f"Acc: {acc:.4f} | ACSA: {acsa:.4f} | F1: {macro_f1:.4f}  ||  "
            f"Best → Acc: {best_acc:.4f} | ACSA: {best_acsa:.4f} | F1: {best_f1:.4f}"
        )

        # ── Log dosyasına yaz ──
        logging.info(
            f"Round {r} | Acc: {acc:.4f} | ACSA: {acsa:.4f} | F1: {macro_f1:.4f} | "
            f"Best Acc: {best_acc:.4f} | Best ACSA: {best_acsa:.4f} | Best F1: {best_f1:.4f}"
        )

        # ── Checkpoint ──
        save_checkpoint(r, global_model.download_params(), metrics_history, checkpoint_dir)

        # ── CSV (tüm metrikler) ──
        result_dir  = f'./results/{args.dataset}'
        os.makedirs(result_dir, exist_ok=True)
        result_file = f'{result_dir}/{args.aggregation_method}_alpha={alpha}.csv'
        n = len(metrics_history['acc'])
        pd.DataFrame({
            'round': list(range(1, n + 1)),
            'acc':   metrics_history['acc'],
            'acsa':  metrics_history['acsa'],
            'f1':    metrics_history['f1'],
        }).to_csv(result_file, index=False, encoding='utf8')


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == '__main__':
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    torch.backends.cudnn.deterministic = True
    args = args_parser()
    main_loop(args.alpha)
