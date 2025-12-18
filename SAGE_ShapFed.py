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


def save_checkpoint(round_num, model_state, fedavg_acc_history, checkpoint_dir, filename='checkpoint.pt'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = {
        'round': round_num,
        'model_state_dict': model_state,
        'fedavg_acc': fedavg_acc_history,
    }
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"\n[SAGE] Checkpoint saved at Round {round_num} to {filepath}")

def load_checkpoint(model, checkpoint_dir, filename='checkpoint.pt'):
    filepath = os.path.join(checkpoint_dir, filename)
    if os.path.exists(filepath):
        print(f"\n[SAGE] Checkpoint found at {filepath}. Loading...")
        try:
            checkpoint = torch.load(filepath, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])
            start_round = checkpoint['round'] + 1
            fedavg_acc = checkpoint['fedavg_acc']
            print(f"[SAGE] Resuming from Round {start_round}. Last Acc: {fedavg_acc[-1] if fedavg_acc else 0:.4f}")
            return start_round, fedavg_acc
        except Exception as e:
            print(f"[SAGE] Error loading checkpoint: {e}. Starting from scratch.")
            return 1, []
    else:
        print("[SAGE] No checkpoint found. Starting from Round 1.")
        return 1, []



def compute_cssv(args, local_models_params, initial_global_params):
    """
    ShapFed: Class-Specific Shapley Values (CSSV) Calculation using Monte Carlo.
    Calculates the Shapley value by summing the marginal contributions 
    (value_of_coalition_plus_i - value_of_coalition_only).
    
    It is expected that 'args' contains the 'shapley_samples' attribute.
    """
    num_clients = len(local_models_params)
    if num_clients == 0:
        return np.array([])
    
    # The names of the last layer parameters in ResNet model
    weight_layer = 'classifier.weight'
    bias_layer = 'classifier.bias'
    
    # 1. Calculate the 'Model Update' (Delta W = Local_Weights - Global_Weights) for each client
    client_updates = []
    for local_params in local_models_params:
        update = {}
        for name in local_params:
            if name in initial_global_params:
                # Calculate the update: W_local - W_global_start
                # Ensure device compatibility for subtraction
                update[name] = local_params[name] - initial_global_params[name].to(local_params[name].device)
        client_updates.append(update)

    shapley_values = np.zeros(num_clients)
    # Get the number of Monte Carlo samples (default to 10 if not in args)
    num_samples = args.shapley_samples if hasattr(args, 'shapley_samples') else 10 
    
    # 2. Monte Carlo Shapley Calculation
    for _ in range(num_samples):
        # Generate a random permutation of clients
        permutation = np.random.permutation(num_clients)
        
        # Calculate the marginal contribution for every client in the permutation
        for i, client_idx in enumerate(permutation):
            
            # Coalition S: Clients preceding the current client in the permutation
            coalition_indices = permutation[:i]
            # Coalition S U {i}: S plus the current client
            coalition_plus_indices = permutation[:i+1]
            
            # Vector of the current client's last-layer update (Delta W_i)
            curr_update_w = torch.cat([
                client_updates[client_idx][weight_layer].view(-1),
                client_updates[client_idx][bias_layer].view(-1)
            ])
            # Normalize for cosine similarity calculation
            curr_update_w_norm = F.normalize(curr_update_w.unsqueeze(0), p=2)

            
            # --- Step 1: Calculate the value of Coalition S (v(S)) ---
            sim_s = 0.0
            if len(coalition_indices) > 0:
                # Calculate the average update of Coalition S (Delta W_S)
                temp_w_s = torch.zeros_like(client_updates[0][weight_layer])
                temp_b_s = torch.zeros_like(client_updates[0][bias_layer])
                
                for c_idx in coalition_indices:
                    temp_w_s += client_updates[c_idx][weight_layer]
                    temp_b_s += client_updates[c_idx][bias_layer]
                
                temp_w_s /= len(coalition_indices)
                temp_b_s /= len(coalition_indices)
                agg_update_s_w = torch.cat([temp_w_s.view(-1), temp_b_s.view(-1)])
                
                # Value function: v(S) = sim(Delta W_S, Delta W_i)
                agg_update_s_w_norm = F.normalize(agg_update_s_w.unsqueeze(0), p=2)
                sim_s = F.cosine_similarity(curr_update_w_norm, agg_update_s_w_norm).item()
            # If S is empty, sim_s remains 0.0

            
            # --- Step 2: Calculate the value of Coalition S U {i} (v(S U {i})) ---
            # Calculate the average update of Coalition S U {i} (Delta W_S U {i})
            temp_w_s_plus_i = torch.zeros_like(client_updates[0][weight_layer])
            temp_b_s_plus_i = torch.zeros_like(client_updates[0][bias_layer])
            
            for c_idx in coalition_plus_indices:
                temp_w_s_plus_i += client_updates[c_idx][weight_layer]
                temp_b_s_plus_i += client_updates[c_idx][bias_layer]
            
            temp_w_s_plus_i /= len(coalition_plus_indices)
            temp_b_s_plus_i /= len(coalition_plus_indices)
            
            agg_update_s_plus_i_w = torch.cat([temp_w_s_plus_i.view(-1), temp_b_s_plus_i.view(-1)])
            
            # Value function: v(S U {i}) = sim(Delta W_{S U {i}}, Delta W_i)
            agg_update_s_plus_i_w_norm = F.normalize(agg_update_s_plus_i_w.unsqueeze(0), p=2)
            sim_s_plus_i = F.cosine_similarity(curr_update_w_norm, agg_update_s_plus_i_w_norm).item()
            
            
            # --- Step 3: Marginal Contribution ---
            # Marginal Contribution = v(S U {i}) - v(S)
            marginal_contribution = sim_s_plus_i - sim_s
            
            # Add the contribution to the client's Shapley value sum
            shapley_values[client_idx] += marginal_contribution

    if num_samples > 0:
        # Average the contributions over all Monte Carlo samples
        shapley_values /= num_samples

    # Normalize the values: Clip negative contributions to zero and divide by the total sum
    # This ensures the resulting weights for aggregation/broadcast are non-negative, as required by ShapFed.
    shapley_values = np.maximum(shapley_values, 0)
    total_shapley = np.sum(shapley_values)
    
    if total_shapley > 0:
        normalized_weights = shapley_values / total_shapley
    else:
        # Fallback: If total contribution is zero, assign equal weights (FedAvg fallback)
        normalized_weights = np.ones(num_clients) / num_clients
            
    return normalized_weights
    

class Global(object):
    def __init__(self, args):
        self.model = ResNet(resnet_size=8, scaling=4,
                            save_activations=False, group_norm_num_groups=None,
                            freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
        self.model.cuda(args.gpu_id)
        self.num_classes = args.num_classes

    def initialize_for_model_fusion(self, args, list_dicts_local_params, list_nums_local_data, initial_global_params):
        """
        Model parameter aggregation.
        - SAGE: weighted average acc. to data quantity (FedAvg).
        - ShapFed: weighted average acc. to Shapley values.
        """
        fused_params = copy.deepcopy(list_dicts_local_params[0])
        
        # define aggregation weights
        if args.aggregation_method == 'ShapFed':
            # ShapFed: contribution-based
            weights = compute_cssv(args, list_dicts_local_params, initial_global_params)
            # Log
            # print(f"ShapFed Weights: {weights}")
        else:
            # Default SAGE (FedAvg): Data size proportional weights
            total_data = sum(list_nums_local_data)
            weights = [n / total_data for n in list_nums_local_data]

        # Weighted Averaging
        for name_param in list_dicts_local_params[0]:
            list_values_param = []
            for dict_local_params, weight in zip(list_dicts_local_params, weights):
                list_values_param.append(dict_local_params[name_param] * weight)
            
            # Aggregate the weighted parameters
            fused_params[name_param] = sum(list_values_param)
            
        return fused_params

    def fedavg_eval(self, fedavg_params, data_test, batch_size_test, args):
        # Evaluation of the global model
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
        # Local training model
        self.local_model = ResNet(resnet_size=8, scaling=4,
                                  save_activations=False, group_norm_num_groups=None,
                                  freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
        # Global/Teacher model for pseudo-label generation in SAGE
        self.local_G = ResNet(resnet_size=8, scaling=4,
                              save_activations=False, group_norm_num_groups=None,
                              freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
        self.local_model.cuda(args.gpu_id)
        self.local_G.cuda(args.gpu_id)
        self.criterion = CrossEntropyLoss().cuda(args.gpu_id)
        self.optimizer = SGD(self.local_model.parameters(), lr=args.lr_local_training, momentum=0.9, weight_decay=1e-4)

    def fixmatch_train(self, args, data_client_labeled, data_client_unlabeled, global_params, r):
        # Dataloaders
        self.labeled_trainloader = DataLoader(
            dataset=data_client_labeled, sampler=RandomSampler(data_client_labeled),
            batch_size=args.batch_size_local_labeled_fixmatch, drop_last=True, num_workers=2, pin_memory=True
        )
        self.unlabeled_trainloader = DataLoader(
            dataset=data_client_unlabeled, sampler=RandomSampler(data_client_unlabeled),
            batch_size=args.batch_size_local_labeled_fixmatch * args.mu, drop_last=True, num_workers=2, pin_memory=True
        )
        # Load global parameters for the current client model and the teacher model
        self.local_model.load_state_dict(global_params)
        self.local_model.train()
        self.local_G.load_state_dict(global_params)
        self.local_G.eval()

        for local_epoch in range(args.local_epochs):
            labeled_iter = iter(self.labeled_trainloader)
            unlabeled_iter = iter(self.unlabeled_trainloader)
            local_iter = int(len(data_client_unlabeled) / args.batch_size_local_labeled_fixmatch)

            for epoch in range(local_iter):
                try: inputs_x, targets_x = labeled_iter.__next__()
                except: labeled_iter = iter(self.labeled_trainloader); inputs_x, targets_x = labeled_iter.__next__()
                try: inputs_u_w, inputs_u_s, targets_u_groundtruth = unlabeled_iter.__next__()
                except: unlabeled_iter = iter(self.unlabeled_trainloader); inputs_u_w, inputs_u_s, targets_u_groundtruth = unlabeled_iter.__next__()

                inputs_x, inputs_u_w, inputs_u_s = inputs_x.cuda(args.gpu_id), inputs_u_w.cuda(args.gpu_id), inputs_u_s.cuda(args.gpu_id)
                batch_size = inputs_x.shape[0]
                
                # FixMatch: Interleave to perform a single forward pass
                inputs = self.interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1).cuda(args.gpu_id)
                targets_x = targets_x.cuda(args.gpu_id)

                _, logits = self.local_model(inputs)
                logits = self.de_interleave(logits, 2 * args.mu + 1)
                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                del logits

                # Loss X(Labeled Loss)
                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

                # Pseudo-Labeling (Global + Local)
                _, logits_u_w_global = self.local_G(inputs_u_w.cuda(args.gpu_id))
                pseudo_label_global = torch.softmax(logits_u_w_global.detach() / args.T, dim=-1)
                max_probs_global, targets_u_global = torch.max(pseudo_label_global, dim=-1)

                pseudo_label_local = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
                max_probs_local, targets_u_local = torch.max(pseudo_label_local, dim=-1)

                targets_u_local_one_hot = F.one_hot(targets_u_local, args.num_classes).float()
                targets_u_global_one_hot = F.one_hot(targets_u_global, args.num_classes).float()

                # Confidence Masks
                mask_local = max_probs_local.ge(args.threshold).float()
                mask_global = max_probs_global.ge(args.threshold).float()

                # SAGE Confidence Discrepancy Logic (Dynamic Weighting)
                delta_c = torch.clamp(torch.abs(max_probs_local - max_probs_global) + 1e-6, min=1e-6, max=1.0)
                kappa = torch.log(torch.tensor(2.0)) / 0.05
                lambda_dynamic = torch.clamp(torch.exp(-kappa * delta_c), min=1e-6, max=1.0)

                # Pseudo-Label Fusion
                final_targets_u = torch.where(
                    mask_local.unsqueeze(1).bool(),
                    lambda_dynamic.unsqueeze(1) * targets_u_local_one_hot + (1 - lambda_dynamic).unsqueeze(1) * targets_u_global_one_hot,
                    targets_u_global_one_hot
                )

                mask_valid = torch.max(mask_local, mask_global)
                logits_u_s_probs = torch.softmax(logits_u_s, dim=-1) + 1e-10
                final_targets_u = final_targets_u + 1e-10
                
                # Loss U (Unlabeled Loss - KL Divergence against soft fused labels)
                Lu = (F.kl_div(logits_u_s_probs.log(), final_targets_u, reduction='none').sum(-1) * mask_valid).mean()

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


def main_loop(alpha):
    args = args_parser()

    # Checkpoint Setup
    checkpoint_dir = os.path.join(args.checkpoint_dir, f'{args.dataset}_a{alpha}_{args.aggregation_method}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")

    # Logging
    log_dir = f'./results/{args.dataset}/logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'SAGE_{args.aggregation_method}_α={alpha}.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file)

    if args.dataset == 'CIFAR10':
        args.num_classes = 10
        args.num_labeled = 500 
        args.num_rounds = 400 
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        data_local_training = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=None)
        data_global_test = datasets.CIFAR10(args.path_cifar10, train=False, transform=transform_test)

    elif args.dataset == 'CIFAR100':
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
        args.num_classes = 10
        args.num_labeled = 460
        args.num_rounds = 300
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        data_local_training = datasets.SVHN(args.path_svhn, split='train', download=True, transform=None)
        data_global_test = datasets.SVHN(args.path_svhn, split='test', transform=transform_test, download=True)

    elif args.dataset == 'CINIC10':
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
        print(f"Dataset {args.dataset} not implemented in this snippet. Please specify one of the following: CIFAR10, CIFAR100, CINIC10 or SVHN.")
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
            num_labeled=args.num_labeled,
            alpha=alpha,
            mu=args.mu,
            num_rounds=args.num_rounds,
            batch_label=args.batch_size_local_labeled,
            batch_unlabel=args.batch_size_local_unlabeled,
        ))
    
    
    # Partitioning
    random_state = np.random.RandomState(args.seed)
    list_label2indices = classify_label(data_local_training, args.num_classes)
    list_label2indices_labeled, list_label2indices_unlabeled = partition_train(list_label2indices, args.num_labeled)

    # IID
    if alpha == 0:
        list_client2indices_labeled = clients_indices_homo(list_label2indices=list_label2indices_labeled, num_classes=args.num_classes, num_clients=args.num_clients)
        list_client2indices_unlabeled = clients_indices_homo(list_label2indices=list_label2indices_unlabeled, num_classes=args.num_classes, num_clients=args.num_clients)
    
    # Non-IID
    else:
        list_client2indices_labeled = clients_indices(list_label2indices=list_label2indices_labeled, num_classes=args.num_classes, num_clients=args.num_clients, non_iid_alpha=alpha, seed=0)
        list_client2indices_unlabeled = clients_indices(list_label2indices=list_label2indices_unlabeled, num_classes=args.num_classes, num_clients=args.num_clients, non_iid_alpha=alpha, seed=0)

    # Show data distribubution
    show_clients_data_distribution(data_local_training, list_client2indices_labeled,
                                   list_client2indices_unlabeled, args.num_classes)

    
    # add labeled samples without labels into the unlabeled dataset
    for client in range(args.num_clients):
        list_client2indices_unlabeled[client].extend(list_client2indices_labeled[client])

    # Model Init
    global_model = Global(args)
    local_model = Local(args)

    # Checkpoint Load
    start_round, fedavg_acc = load_checkpoint(global_model.model, checkpoint_dir)

    total_clients = list(range(args.num_clients))
    indices2data_labeled = Indices2Dataset_labeled(data_local_training)
    indices2data_unlabeled = Indices2Dataset_unlabeled_fixmatch(data_local_training)

    # Training Loop
    for r in tqdm(range(start_round, args.num_rounds + 1), desc='Server'):
        
        # download current global parameters (necessary in ShapFed for update calculation)
        dict_global_params = global_model.download_params()

        online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
        list_dicts_local_params = []
        list_nums_local_data = []

        # Local Training
        for client in online_clients:
            indices2data_labeled.load(list_client2indices_labeled[client])
            data_client_labeled = indices2data_labeled
            indices2data_unlabeled.load(list_client2indices_unlabeled[client])
            data_client_unlabeled = indices2data_unlabeled

            list_nums_local_data.append(len(data_client_labeled) + len(data_client_unlabeled))
            
            # SAGE Training
            local_params = local_model.fixmatch_train(args, data_client_labeled, data_client_unlabeled, copy.deepcopy(dict_global_params), r)
            list_dicts_local_params.append(copy.deepcopy(local_params))

        # AGGREGATION: SAGE vs ShapFed
        fedavg_params = global_model.initialize_for_model_fusion(args, list_dicts_local_params, list_nums_local_data, dict_global_params)

        # Global Model update
        global_model.model.load_state_dict(fedavg_params)

        # Evaluate
        global_acc = global_model.fedavg_eval(copy.deepcopy(fedavg_params), data_global_test, args.batch_size_test, args)
        fedavg_acc.append(global_acc)
        print(f'Round {r} Accuracy: {global_acc}')

        # Save Checkpoint (after each round)
        save_checkpoint(r, global_model.download_params(), fedavg_acc, checkpoint_dir)

        # CSV Save
        result_dir = f'./results/{args.dataset}'
        os.makedirs(result_dir, exist_ok=True)
        result_file = f'{result_dir}/{args.aggregation_method}_alpha={alpha}.csv'
        acc_df = pd.DataFrame({'acc': fedavg_acc}, index=list(range(1, len(fedavg_acc) + 1)))
        acc_df.to_csv(result_file, encoding='utf8')

if __name__ == '__main__':
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    torch.backends.cudnn.deterministic = True

    args = args_parser()
    main_loop(args.alpha)
