import os
import copy
import random
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import recall_score, f1_score
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

from Model.resnet import ResNet
from options import args_parser
from Dataset.dataset import classify_label, show_clients_data_distribution, Indices2Dataset_labeled, Indices2Dataset_unlabeled_fixmatch, partition_train
from Dataset.sample_dirichlet import clients_indices, clients_indices_homo
from tqdm import tqdm

def save_checkpoint(round_num, model_state, fedavg_acc, checkpoint_dir, filename='checkpoint.pt'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = {'round': round_num, 'model_state_dict': model_state, 'fedavg_acc': fedavg_acc}
    torch.save(state, os.path.join(checkpoint_dir, filename))

def load_checkpoint(model, checkpoint_dir, filename='checkpoint.pt'):
    filepath = os.path.join(checkpoint_dir, filename)
    if os.path.exists(filepath):
        print(f"\n[SAGE] Checkpoint found at {filepath}. Loading...")
        checkpoint = torch.load(filepath, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint['round'] + 1, checkpoint['fedavg_acc']
    return 1, []

def compute_cssv(args, local_models_params, initial_global_params):
    num_clients = len(local_models_params)
    if num_clients == 0: return np.array([])
    weight_layer, bias_layer = 'classifier.weight', 'classifier.bias'
    
    client_updates = []
    for local_params in local_models_params:
        update = {}
        for name in local_params:
            if name in initial_global_params:
                update[name] = local_params[name] - initial_global_params[name].to(local_params[name].device)
        client_updates.append(update)

    shapley_values = np.zeros((num_clients, args.num_classes))
    num_samples = getattr(args, 'shapley_samples', 10)

    for _ in range(num_samples):
        permutation = np.random.permutation(num_clients)
        for i, client_idx in enumerate(permutation):
            coalition_indices = permutation[:i]
            coalition_plus_indices = permutation[:i+1]
            
            for c in range(args.num_classes):
                curr_w_c = torch.cat([client_updates[client_idx][weight_layer][c].view(-1), client_updates[client_idx][bias_layer][c].view(-1)])
                curr_w_norm = curr_w_c.unsqueeze(0) if torch.norm(curr_w_c) == 0 else F.normalize(curr_w_c.unsqueeze(0), p=2)

                sim_s = 0.0
                if len(coalition_indices) > 0:
                    tw_c = sum([client_updates[co_idx][weight_layer][c] for co_idx in coalition_indices]) / len(coalition_indices)
                    tb_c = sum([client_updates[co_idx][bias_layer][c] for co_idx in coalition_indices]) / len(coalition_indices)
                    agg_c = torch.cat([tw_c.view(-1), tb_c.view(-1)])
                    if torch.norm(agg_c) > 0 and torch.norm(curr_w_c) > 0:
                        sim_s = F.cosine_similarity(curr_w_norm, F.normalize(agg_c.unsqueeze(0), p=2)).item()

                tw2_c = sum([client_updates[co_idx][weight_layer][c] for co_idx in coalition_plus_indices]) / len(coalition_plus_indices)
                tb2_c = sum([client_updates[co_idx][bias_layer][c] for co_idx in coalition_plus_indices]) / len(coalition_plus_indices)
                agg2_c = torch.cat([tw2_c.view(-1), tb2_c.view(-1)])
                sim_si = 0.0
                if torch.norm(agg2_c) > 0 and torch.norm(curr_w_c) > 0:
                    sim_si = F.cosine_similarity(curr_w_norm, F.normalize(agg2_c.unsqueeze(0), p=2)).item()

                shapley_values[client_idx, c] += (sim_si - sim_s)

    if num_samples > 0: shapley_values /= num_samples
    shapley_values = np.maximum(shapley_values, 0)
    for c in range(args.num_classes):
        col_sum = np.sum(shapley_values[:, c])
        if col_sum > 0: shapley_values[:, c] /= col_sum
        else: shapley_values[:, c] = 1.0 / num_clients
    return shapley_values

def patch_resnet_for_ham(model):
    model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    return model

class Global(object):
    def __init__(self, args):
        self.model = ResNet(resnet_size=8, scaling=4, save_activations=False, group_norm_num_groups=None, freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
        if args.dataset == 'HAM10000': patch_resnet_for_ham(self.model)
        self.model.cuda(args.gpu_id)
        self.num_classes = args.num_classes

    def initialize_for_model_fusion(self, args, list_dicts_local_params, list_nums_local_data, initial_global_params):
        if args.aggregation_method == 'FedAvg':
            # ORİJİNAL FEDAVG MATEMATİĞİ - PÜRÜZSÜZLÜĞÜN SIRRI BURASI
            fedavg_global_params = copy.deepcopy(list_dicts_local_params[0])
            for name_param in list_dicts_local_params[0]:
                list_values_param = []
                for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
                    list_values_param.append(dict_local_params[name_param] * num_local_data)
                
                fused_tensor = sum(list_values_param) / sum(list_nums_local_data)
                
                if list_dicts_local_params[0][name_param].dtype == torch.int64:
                    fedavg_global_params[name_param] = fused_tensor.to(torch.int64)
                else:
                    fedavg_global_params[name_param] = fused_tensor
            return fedavg_global_params

        else:
            # SHAPFED LOGIC
            fused_params = copy.deepcopy(list_dicts_local_params[0])
            num_clients = len(list_dicts_local_params)
            cssv_weights = compute_cssv(args, list_dicts_local_params, initial_global_params)
            client_avg_weights = np.mean(cssv_weights, axis=1)
            if np.sum(client_avg_weights) > 0: client_avg_weights /= np.sum(client_avg_weights)
            else: client_avg_weights = np.ones(num_clients) / num_clients

            for name_param in list_dicts_local_params[0]:
                if name_param == 'classifier.weight' or name_param == 'classifier.bias':
                    fused_tensor = torch.zeros_like(list_dicts_local_params[0][name_param], dtype=torch.float32)
                    for c in range(args.num_classes):
                        for i in range(num_clients):
                            fused_tensor[c] += list_dicts_local_params[i][name_param][c] * cssv_weights[i, c]
                else:
                    fused_tensor = sum(list_dicts_local_params[i][name_param] * client_avg_weights[i] for i in range(num_clients))
                
                if list_dicts_local_params[0][name_param].dtype == torch.int64:
                    fused_params[name_param] = fused_tensor.to(torch.int64)
                else:
                    fused_params[name_param] = fused_tensor
            return fused_params

    def fedavg_eval(self, fedavg_params, data_test, batch_size_test, args):
        self.model.load_state_dict(fedavg_params)
        self.model.eval()
        num_corrects = 0
        with torch.no_grad():
            test_loader = DataLoader(data_test, batch_size_test)
            for images, labels in test_loader:
                images, labels = images.cuda(args.gpu_id), labels.cuda(args.gpu_id)
                _, outputs = self.model(images)
                _, predicts = torch.max(outputs, -1)
                num_corrects += torch.sum(torch.eq(predicts.cpu(), labels.cpu())).item()
        accuracy = num_corrects / len(data_test)
        torch.cuda.empty_cache()
        return accuracy

    def download_params(self):
        return self.model.state_dict()

class Local(object):
    def __init__(self, args):
        self.local_model = ResNet(resnet_size=8, scaling=4, save_activations=False, group_norm_num_groups=None, freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
        self.local_G = ResNet(resnet_size=8, scaling=4, save_activations=False, group_norm_num_groups=None, freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
        if args.dataset == 'HAM10000':
            patch_resnet_for_ham(self.local_model); patch_resnet_for_ham(self.local_G)
        self.local_model.cuda(args.gpu_id); self.local_G.cuda(args.gpu_id)
        self.criterion = CrossEntropyLoss().cuda(args.gpu_id)
        self.optimizer = SGD(self.local_model.parameters(), lr=args.lr_local_training, momentum=0.9, weight_decay=1e-4)

    def fixmatch_train(self, args, data_client_labeled, data_client_unlabeled, global_params, r):
        self.labeled_trainloader = DataLoader(data_client_labeled, sampler=RandomSampler(data_client_labeled), batch_size=args.batch_size_local_labeled_fixmatch, drop_last=True, num_workers=2, pin_memory=True)
        self.unlabeled_trainloader = DataLoader(data_client_unlabeled, sampler=RandomSampler(data_client_unlabeled), batch_size=args.batch_size_local_labeled_fixmatch * args.mu, drop_last=True, num_workers=2, pin_memory=True)
        
        self.local_model.load_state_dict(global_params); self.local_model.train()
        self.local_G.load_state_dict(global_params); self.local_G.eval()

        for local_epoch in range(args.local_epochs):
            labeled_iter = iter(self.labeled_trainloader)
            unlabeled_iter = iter(self.unlabeled_trainloader)
            local_iter = int(len(data_client_unlabeled) / args.batch_size_local_labeled_fixmatch)

            for epoch in range(local_iter):
                try: inputs_x, targets_x = next(labeled_iter)
                except StopIteration: labeled_iter = iter(self.labeled_trainloader); inputs_x, targets_x = next(labeled_iter)
                try: inputs_u_w, inputs_u_s, targets_u_groundtruth = next(unlabeled_iter)
                except StopIteration: unlabeled_iter = iter(self.unlabeled_trainloader); inputs_u_w, inputs_u_s, targets_u_groundtruth = next(unlabeled_iter)

                inputs_x, inputs_u_w, inputs_u_s = inputs_x.cuda(args.gpu_id), inputs_u_w.cuda(args.gpu_id), inputs_u_s.cuda(args.gpu_id)
                batch_size = inputs_x.shape[0]
                inputs = self.interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1).cuda(args.gpu_id)
                targets_x = targets_x.cuda(args.gpu_id)

                _, logits = self.local_model(inputs)
                logits = self.de_interleave(logits, 2 * args.mu + 1)
                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                del logits

                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

                _, logits_u_w_global = self.local_G(inputs_u_w.cuda(args.gpu_id))
                pseudo_label_global = torch.softmax(logits_u_w_global.detach() / args.T, dim=-1)
                max_probs_global, targets_u_global = torch.max(pseudo_label_global, dim=-1)

                pseudo_label_local = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
                max_probs_local, targets_u_local = torch.max(pseudo_label_local, dim=-1)

                targets_u_local_one_hot = F.one_hot(targets_u_local, args.num_classes).float()
                targets_u_global_one_hot = F.one_hot(targets_u_global, args.num_classes).float()

                mask_local = max_probs_local.ge(args.threshold).float()
                mask_global = max_probs_global.ge(args.threshold).float()

                delta_c = torch.clamp(torch.abs(max_probs_local - max_probs_global) + 1e-6, min=1e-6, max=1.0)
                kappa = torch.log(torch.tensor(2.0)) / 0.05
                lambda_dynamic = torch.clamp(torch.exp(-kappa * delta_c), min=1e-6, max=1.0)

                final_targets_u = torch.where(
                    mask_local.unsqueeze(1).bool(),
                    lambda_dynamic.unsqueeze(1) * targets_u_local_one_hot + (1-lambda_dynamic).unsqueeze(1) * targets_u_global_one_hot,
                    targets_u_global_one_hot
                )

                mask_valid = torch.max(mask_local, mask_global)
                logits_u_s_probs = torch.softmax(logits_u_s, dim=-1) + 1e-10
                final_targets_u = final_targets_u + 1e-10
                Lu = (F.kl_div(logits_u_s_probs.log(), final_targets_u, reduction='none').sum(-1) * mask_valid).mean()

                loss = Lx + args.lambda_u * Lu

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
        final_state = {k: v.cpu() for k, v in self.local_model.state_dict().items()}
        self.optimizer.zero_grad(set_to_none=True)
        return final_state

    def interleave(self, x, size):
        s = list(x.shape)
        return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

    def de_interleave(self, x, size):
        s = list(x.shape)
        return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

class _SubsetImageFolder(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices, transform=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
        max_len = len(base_dataset)
        self.valid_indices = [i for i in indices if i < max_len]
        self.targets = [base_dataset.targets[i] for i in self.valid_indices]

    def __len__(self): return len(self.valid_indices)
    def __getitem__(self, idx):
        try:
            img, label = self.base_dataset[self.valid_indices[idx]]
            if self.transform: img = self.transform(img)
            return img, label
        except:
            return self.__getitem__((idx + 1) % len(self.valid_indices))

def main_loop(alpha):
    args = args_parser()
    checkpoint_dir = os.path.join(args.checkpoint_dir, f'{args.dataset}_a{alpha}_{args.aggregation_method}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    log_dir = f'./results/{args.dataset}/logs'
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename=os.path.join(log_dir, f'SAGE_{args.aggregation_method}_a={alpha}.log'), filemode='a')

    if args.dataset == 'CIFAR10':
        args.num_classes = 10; args.num_labeled = 500; args.num_rounds = 400
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])
        
        # Orijinal SAGE Hızı İçin Direkt Yükleme (Wrapper Yok)
        data_local_training = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=None)
        data_global_test = datasets.CIFAR10(args.path_cifar10, train=False, download=True, transform=transform_test)

    elif args.dataset == 'HAM10000':
        args.num_classes = 7; args.num_labeled = 100; args.num_rounds = 400
        transform_test = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.763, 0.545, 0.570), (0.140, 0.152, 0.169))])
        full_dataset = ImageFolder(root=args.path_ham10000, transform=None)
        all_indices = list(range(len(full_dataset)))
        train_indices, test_indices = train_test_split(all_indices, test_size=0.20, stratify=full_dataset.targets, random_state=args.seed)
        data_local_training = _SubsetImageFolder(full_dataset, train_indices)
        data_global_test = _SubsetImageFolder(ImageFolder(root=args.path_ham10000, transform=transform_test), test_indices)
    else:
        exit(1)

    random_state = np.random.RandomState(args.seed)
    list_label2indices = classify_label(data_local_training, args.num_classes)
    
    # IPC hesaplaması: Sınıf başına etiket
    ipc = args.num_labeled
    list_label2indices_labeled, list_label2indices_unlabeled = partition_train(list_label2indices, ipc)

    if alpha == 0:
        list_client2indices_labeled = clients_indices_homo(list_label2indices_labeled, args.num_classes, args.num_clients)
        list_client2indices_unlabeled = clients_indices_homo(list_label2indices_unlabeled, args.num_classes, args.num_clients)
    else:
        list_client2indices_labeled = clients_indices(list_label2indices_labeled, args.num_classes, args.num_clients, alpha, seed=0)
        list_client2indices_unlabeled = clients_indices(list_label2indices_unlabeled, args.num_classes, args.num_clients, alpha, seed=0)

    for client in range(args.num_clients):
        list_client2indices_unlabeled[client].extend(list_client2indices_labeled[client])

    global_model = Global(args)
    local_model = Local(args)

    start_round, fedavg_acc = load_checkpoint(global_model.model, checkpoint_dir)

    indices2data_labeled = Indices2Dataset_labeled(data_local_training, args.dataset)
    indices2data_unlabeled = Indices2Dataset_unlabeled_fixmatch(data_local_training, args.dataset)

    total_clients = list(range(args.num_clients))

    for r in tqdm(range(start_round, args.num_rounds + 1), desc='Server'):
        dict_global_params = global_model.download_params()
        online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
        
        list_dicts_local_params = []
        list_nums_local_data = []

        for client in online_clients:
            indices2data_labeled.load(list_client2indices_labeled[client])
            indices2data_unlabeled.load(list_client2indices_unlabeled[client])
            
            # ORİJİNAL FEDAVG AĞIRLIĞI İÇİN UZUNLUKLAR
            list_nums_local_data.append(len(indices2data_labeled) + len(indices2data_unlabeled))
            
            local_params = local_model.fixmatch_train(args, indices2data_labeled, indices2data_unlabeled, copy.deepcopy(dict_global_params), r)
            list_dicts_local_params.append(local_params)

        fedavg_params = global_model.initialize_for_model_fusion(args, list_dicts_local_params, list_nums_local_data, dict_global_params)
        global_model.model.load_state_dict(fedavg_params)

        acc = global_model.fedavg_eval(copy.deepcopy(fedavg_params), data_global_test, args.batch_size_test, args)
        fedavg_acc.append(acc)
        
        print(f"\n[Round {r}] Accuracy: {acc:.4f} | Best: {max(fedavg_acc):.4f}")
        logging.info(f"Round {r} | Acc: {acc:.4f}")

        save_checkpoint(r, global_model.download_params(), fedavg_acc, checkpoint_dir)

        result_dir = f'./results/{args.dataset}'
        os.makedirs(result_dir, exist_ok=True)
        pd.DataFrame({'acc': fedavg_acc}, index=list(range(1, len(fedavg_acc) + 1))).to_csv(f'{result_dir}/{args.aggregation_method}_a={alpha}.csv')

if __name__ == '__main__':
    torch.manual_seed(7); torch.cuda.manual_seed(7); np.random.seed(7); random.seed(7)
    torch.backends.cudnn.deterministic = True
    args = args_parser()
    main_loop(args.alpha)
