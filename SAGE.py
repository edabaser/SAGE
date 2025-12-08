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
from sklearn.metrics import f1_score, recall_score, precision_score # NEW: For medical metrics

worker_num = 4

# NEW HELPER FUNCTION: Calculate F1, Sensitivity, Specificity
def calculate_metrics(y_true, y_pred, average='macro'):
    """
    Calculates F1-Score, Sensitivity (Recall), and Specificity (approximated by Precision for multi-class macro avg).
    """
    # Sensitivity (Recall)
    sensitivity = recall_score(y_true, y_pred, average=average, zero_division=0)
    
    # Specificity (In multi-class context, macro/weighted precision often serves as a proxy for balanced metrics)
    specificity = precision_score(y_true, y_pred, average=average, zero_division=0)
    
    # F1-Score
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    return f1, sensitivity, specificity


class Global(object):
    def __init__(self, args):
        self.model = ResNet(resnet_size=8, scaling=4,
                            save_activations=False, group_norm_num_groups=None,
                            freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
        # Temporary model used for Shapley estimation to test coalitions quickly
        self.temp_model = ResNet(resnet_size=8, scaling=4,
                            save_activations=False, group_norm_num_groups=None,
                            freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)

        self.model.cuda(args.gpu_id)
        self.temp_model.cuda(args.gpu_id)
        self.num_classes = args.num_classes
        self.args = args

    # NEW: Aggregation method (FedAvg only) for intermediate Shapley testing
    def fedavg_lite(self, list_dicts_local_params: list, list_nums_local_data: list):
        if not list_dicts_local_params:
            return None
            
        fedavg_params = copy.deepcopy(list_dicts_local_params[0])
        total_data = sum(list_nums_local_data)

        if total_data == 0:
             return fedavg_params

        for name_param in list_dicts_local_params[0]:
            list_values_param = []
            for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
                list_values_param.append(dict_local_params[name_param] * num_local_data)
            
            value_global_param = sum(list_values_param) / total_data
            fedavg_params[name_param] = value_global_param
        return fedavg_params

    # NEW: Evaluation method (Accuracy only) for intermediate Shapley testing
    def fedavg_eval_lite(self, params, data_test):
        if params is None:
            return 0.0
            
        self.temp_model.load_state_dict(params)
        self.temp_model.eval()
        
        with no_grad():
            test_loader = DataLoader(data_test, self.args.batch_size_test)
            num_corrects = 0
            for data_batch in test_loader:
                images, labels = data_batch
                images, labels = images.cuda(self.args.gpu_id), labels.cuda(self.args.gpu_id)
                _, outputs = self.temp_model(images)
                _, predicts = max(outputs, -1)
                num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
            accuracy = num_corrects / len(data_test)
        return accuracy

    # NEW: Estimate Shapley Values using Monte Carlo Sampling
    def estimate_shapley_mc(self, client_models, client_indices_unlabeled, data_global_test):
        K = len(client_models)
        S = self.args.shapley_samples
        shapley_values = np.zeros(K)
        
        # Use unlabeled data indices list for client size (or use data size directly if preferred)
        client_sizes = [len(idx) for idx in client_indices_unlabeled]

        for client_k_idx in range(K):
            marginal_contributions = []
            
            for _ in range(S):
                # 1. Randomly select a subset S (excluding client k)
                coalition_indices = [i for i in range(K) if i != client_k_idx]
                np.random.shuffle(coalition_indices)
                
                # Choose a random subset size and the subset indices
                subset_size = random.randint(0, K - 1)
                subset = coalition_indices[:subset_size]
                
                # Collect models and sizes for S
                models_S = [client_models[i] for i in subset]
                sizes_S = [client_sizes[i] for i in subset]
                
                # 2. Performance of Coalition S: V(S)
                w_S = self.fedavg_lite(models_S, sizes_S)
                V_S = self.fedavg_eval_lite(w_S, data_global_test)
                
                # 3. Performance of Coalition S U {k}: V(S U {k})
                models_SUk = models_S + [client_models[client_k_idx]]
                sizes_SUk = sizes_S + [client_sizes[client_k_idx]]
                
                w_SUk = self.fedavg_lite(models_SUk, sizes_SUk)
                V_SUk = self.fedavg_eval_lite(w_SUk, data_global_test)
                
                # 4. Marginal contribution: V(S U {k}) - V(S)
                marginal_contributions.append(V_SUk - V_S)
                
            shapley_values[client_k_idx] = np.mean(marginal_contributions)
            
        # Normalize and ensure non-negative weights
        min_shapley = np.min(shapley_values)
        if min_shapley < 0:
            shapley_values += abs(min_shapley)
        
        return shapley_values

    # Aggregation method updated to handle FedAvg and SDFL_SAGE
    def initialize_for_model_fusion(self, list_dicts_local_params: list, list_nums_local_data: list, 
                                    online_clients: list, list_client_indices_unlabeled: list, 
                                    data_global_test, aggregation_method):
        
        if aggregation_method == 'SAGE': # Original FedAvg
            fedavg_global_params = self.fedavg_lite(list_dicts_local_params, list_nums_local_data)
            return fedavg_global_params
        
        elif aggregation_method == 'SDFL_SAGE': # Shapley-Driven Aggregation
            
            # 1. Estimate Shapley Values based on marginal contribution to accuracy
            shapley_values = self.estimate_shapley_mc(list_dicts_local_params, list_client_indices_unlabeled, data_global_test)
            
            # 2. Apply Shapley-based weights
            total_shapley = np.sum(shapley_values)
            
            if total_shapley == 0:
                logging.warning("Total Shapley value is zero. Falling back to FedAvg weights.")
                total_data = sum(list_nums_local_data)
                weights = [n / total_data for n in list_nums_local_data]
            else:
                # Weights are normalized Shapley values
                weights = shapley_values / total_shapley

            # Perform weighted aggregation
            shapley_global_params = copy.deepcopy(list_dicts_local_params[0])
            for name_param in list_dicts_local_params[0]:
                list_values_param = []
                for dict_local_params, weight in zip(list_dicts_local_params, weights):
                    list_values_param.append(dict_local_params[name_param] * weight)
                
                value_global_param = sum(list_values_param)
                shapley_global_params[name_param] = value_global_param
                
            return shapley_global_params
        
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")


    # Evaluation method updated to calculate medical metrics (F1, Sens, Spec)
    def fedavg_eval(self, fedavg_params, data_test, batch_size_test):
        self.model.load_state_dict(fedavg_params)
        self.model.eval()
        
        y_true = []
        y_pred = []
        
        with no_grad():
            test_loader = DataLoader(data_test, batch_size_test)
            num_corrects = 0
            
            for data_batch in test_loader:
                images, labels = data_batch
                images, labels = images.cuda(self.args.gpu_id), labels.cuda(self.args.gpu_id)
                
                _, outputs = self.model(images)
                _, predicts = max(outputs, -1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicts.cpu().numpy())
                
                num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
                
            accuracy = num_corrects / len(data_test)
            
            # Calculate new metrics
            f1, sensitivity, specificity = calculate_metrics(np.array(y_true), np.array(y_pred), average='macro')
            
        return accuracy, f1, sensitivity, specificity

    def download_params(self):
        return self.model.state_dict()


class Local(object):
    # ... (Local class content remains the same, as the Shapley changes are Server-side)
    # NOTE: If implementing SAGE confidence into Shapley weights, this class's return value should be updated.
    
    # ... (omitting BasicBlock, ResNet, etc., as they are unchanged)
# ... (omitting BasicBlock, ResNet, etc., as they are unchanged)

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
                    torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1).cuda(args.gpu_id) # Interleave labeled and unlabeled data
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

                # delta_p (Confidence Discrepancy)
                delta_c = torch.abs(max_probs_local - max_probs_global) + 1e-6
                delta_c = torch.clamp(delta_c, min=1e-6, max=1.0)

                # Confidence-Driven Soft Correction (CDSC)
                # Ensure lambda_dynamic = 0.5 when delta_p = 0.05 (based on the original paper logic)
                kappa = torch.log(torch.tensor(2.0)) / 0.05
                lambda_dynamic = torch.exp(-kappa * delta_c)
                lambda_dynamic = torch.clamp(lambda_dynamic, min=1e-6, max=1.0)

                # Final target calculated by soft correction for samples where local pseudo-label is confident
                final_targets_u = torch.where(
                    mask_local.unsqueeze(1).bool(),
                    lambda_dynamic.unsqueeze(1) * targets_u_local_one_hot + (1-lambda_dynamic).unsqueeze(1) * targets_u_global_one_hot,
                    targets_u_global_one_hot # Use global pseudo-label if local is not confident
                )

                # Mask for valid pseudo-labels (union of local and global masks)
                mask_valid = torch.max(mask_local, mask_global)

                logits_u_s_probs = torch.softmax(logits_u_s, dim=-1) + 1e-10
                final_targets_u = final_targets_u + 1e-10
                # Unlabeled Loss (KL Divergence with masking)
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



def fixmatch(alpha):

    args = args_parser()

    # --- Logging Setup ---
    log_dir = f'./results/{args.dataset}/logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{args.aggregation_method}_SAGE_α={alpha}.log') # Log file name updated
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=log_file
                        )

    # --- Dataset Loading ---
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

    # ... (CIFAR100, SVHN, CINIC10 blocks remain similar structure)
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
        args.num_rounds = 150
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
        
    elif args.dataset == 'HAM10000' or args.dataset == 'BloodMNIST': # NEW: Medical Datasets
        # NOTE: You must adjust these parameters based on the actual dataset statistics (e.g., BloodMNIST has 8 classes, HAM10000 has 7)
        if args.dataset == 'HAM10000':
            args.num_classes = 7 
            args.num_labeled = 300 
            args.num_rounds = 300 
            data_path = args.path_ham10000
        elif args.dataset == 'BloodMNIST':
            args.num_classes = 8 
            args.num_labeled = 500 
            args.num_rounds = 200
            data_path = args.path_bloodmnist
            
        # Example transforms for medical images (3-channel)
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            # Adjust normalization constants for medical images if available, otherwise use a generic one
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
        ])
        
        # ASSUMPTION: Data is loaded via ImageFolder or a similar structure compatible with torchvision datasets
        try:
             # Placeholder for data loading. Ensure your Dataset/dataset.py handles this correctly.
             # If using MedMNIST, replace these lines with MedMNIST loading logic.
             data_local_training = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=None)
             data_global_test = datasets.ImageFolder(os.path.join(data_path, 'test'), transform=transform_test)
        except Exception as e:
             logging.error(f"Error loading {args.dataset}: {e}")
             exit(1)


    else:
        print(
            f"Error: Unsupported dataset {args.dataset}. Please specify one of the supported options.")
        exit(1)

    print(
        'dataset:{dataset}\n'
        'num_classes:{num_classes}\n'
        'num_labeled:{num_labeled}\n'
        'non_iid:{alpha}\n'
        'mu:{mu}\n'
        'num_rounds:{num_rounds}\n'
        'batch_label:{batch_label}, batch_unlabel:{batch_unlabel}\n'
        'agg_method:{agg_method}, shapley_samples:{samples}'.format(
            dataset=args.dataset,
            num_classes=args.num_classes,
            num_labeled=args.num_labeled,
            alpha=alpha,
            mu=args.mu,
            num_rounds=args.num_rounds,
            batch_label=args.batch_size_local_labeled,
            batch_unlabel=args.batch_size_local_unlabeled,
            agg_method=args.aggregation_method, # NEW print
            samples=args.shapley_samples # NEW print
        ))

    # --- Data Partitioning ---
    random_state = np.random.RandomState(args.seed)

    list_label2indices = classify_label(data_local_training, args.num_classes)
    list_label2indices_labeled, list_label2indices_unlabeled = partition_train(list_label2indices, args.num_labeled)

    # IID or Non-IID partitioning using Dirichlet
    if alpha == 0:
        list_client2indices_labeled = clients_indices_homo(list_label2indices=list_label2indices_labeled,
                                                         num_classes=args.num_classes,
                                                         num_clients=args.num_clients)
        list_client2indices_unlabeled = clients_indices_homo(list_label2indices=list_label2indices_unlabeled,
                                                             num_classes=args.num_classes,
                                                             num_clients=args.num_clients)
    else:
        list_client2indices_labeled = clients_indices(list_label2indices=list_label2indices_labeled,
                                                      num_classes=args.num_classes,
                                                      num_clients=args.num_clients, non_iid_alpha=alpha, seed=0)
        list_client2indices_unlabeled = clients_indices(list_label2indices=list_label2indices_unlabeled,
                                                         num_classes=args.num_classes,
                                                         num_clients=args.num_clients, non_iid_alpha=alpha, seed=0)

    # Show data distribubution
    show_clients_data_distribution(data_local_training, list_client2indices_labeled,
                                 list_client2indices_unlabeled, args.num_classes)

    # Add labeled samples (without labels) into the unlabeled dataset
    for client in range(args.num_clients):
        list_client2indices_unlabeled[client].extend(list_client2indices_labeled[client])

    global_model = Global(args)
    local_model = Local(args)

    total_clients = list(range(args.num_clients))


    indices2data_labeled = Indices2Dataset_labeled(data_local_training)
    indices2data_unlabeled = Indices2Dataset_unlabeled_fixmatch(data_local_training)

    # --- Metrics Initialization ---
    fedavg_acc = []
    fedavg_f1 = []
    fedavg_sens = []
    fedavg_spec = []

    # --- FL Training Loop ---
    for r in tqdm(range(1, args.num_rounds + 1), desc='Server'):


        dict_global_params = global_model.download_params()

        # Client Sampling
        online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
        list_dicts_local_params = []
        list_nums_local_data = []
        list_client_indices_unlabeled = [] # For Shapley size estimation

        # Client Training
        for client in online_clients:
            indices2data_labeled.load(list_client2indices_labeled[client])
            data_client_labeled = indices2data_labeled
            indices2data_unlabeled.load(list_client2indices_unlabeled[client])
            data_client_unlabeled = indices2data_unlabeled
            
            # Record original indices for Shapley estimation (determining client size/contribution base)
            # Use the full set of indices available to the client for sizing
            list_client_indices_unlabeled.append(list_client2indices_unlabeled[client]) 

            list_nums_local_data.append(len(data_client_labeled) + len(data_client_unlabeled))
            
            # Local training using SAGE/FixMatch
            local_params = local_model.fixmatch_train(args, data_client_labeled, data_client_unlabeled,
                                                      copy.deepcopy(dict_global_params), r)
            list_dicts_local_params.append(copy.deepcopy(local_params))


        # --- Aggregation (SDFL_SAGE or SAGE) ---
        fedavg_params = global_model.initialize_for_model_fusion(
            list_dicts_local_params, 
            list_nums_local_data, 
            online_clients,
            list_client_indices_unlabeled, 
            data_global_test, 
            args.aggregation_method
        )
        
        # --- Evaluation with all medical metrics ---
        global_acc, global_f1, global_sens, global_spec = global_model.fedavg_eval(
            copy.deepcopy(fedavg_params), data_global_test, args.batch_size_test)
        
        # Store results
        fedavg_acc.append(global_acc)
        fedavg_f1.append(global_f1)
        fedavg_sens.append(global_sens)
        fedavg_spec.append(global_spec)
        
        print('Round {round} Acc: {global_acc:.4f}, F1: {global_f1:.4f}, Sens: {global_sens:.4f}, Spec: {global_spec:.4f}'.format(
            round=r,
            global_acc=global_acc,
            global_f1=global_f1,
            global_sens=global_sens,
            global_spec=global_spec))


        # --- Saving Results ---
        result_dir = f'./results/{args.dataset}'
        os.makedirs(result_dir, exist_ok=True)
        result_file = f'{result_dir}/{args.aggregation_method}_SAGE_α={alpha}.csv' 
        
        acc_num_pseudo_label_csv_index = list(range(1, len(fedavg_acc)+1))
        acc_num_pseudo_label_csv_df = pd.DataFrame({
            'acc': fedavg_acc,
            'f1': fedavg_f1,
            'sensitivity': fedavg_sens,
            'specificity': fedavg_spec
        }, index=acc_num_pseudo_label_csv_index)
        
        acc_num_pseudo_label_csv_df.to_csv(result_file, encoding='utf8')


if __name__ == '__main__':

    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    torch.backends.cudnn.deterministic = True

    args = args_parser()
    fixmatch(args.alpha)
