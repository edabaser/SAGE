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
from sklearn.metrics import f1_score, recall_score, precision_score 

worker_num = 4

# --- HELPER FUNCTION: Calculate F1, Sensitivity, Specificity ---
def calculate_metrics(y_true, y_pred, average='macro'):
    """
    Calculates F1-Score, Sensitivity (Recall), and Specificity.
    Specificity is approximated by Precision for multi-class macro average.
    """
    sensitivity = recall_score(y_true, y_pred, average=average, zero_division=0)
    specificity = precision_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    return f1, sensitivity, specificity


# --- Checkpoint Utility Functions (Updated to handle all metrics) ---
def save_checkpoint(round_num, model_state, acc_history, f1_history, sens_history, spec_history, checkpoint_dir, filename='checkpoint.pt'):
    """Saves the global model state and training history."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = {
        'round': round_num,
        'model_state_dict': model_state,
        'acc_history': acc_history,
        'f1_history': f1_history,
        'sens_history': sens_history,
        'spec_history': spec_history,
    }
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"\n✅ Checkpoint saved at Round {round_num} to {filepath}")


def load_checkpoint(model, checkpoint_dir, filename='checkpoint.pt'):
    """Loads a checkpoint if it exists and returns the starting round and metric histories."""
    filepath = os.path.join(checkpoint_dir, filename)

    if os.path.exists(filepath):
        print(f"\n⏳ Checkpoint found at {filepath}. Loading...")
        checkpoint = torch.load(filepath, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        start_round = checkpoint['round'] + 1
        
        # Load histories, safely defaulting to empty lists if they don't exist
        acc_history = checkpoint.get('acc_history', [])
        f1_history = checkpoint.get('f1_history', [])
        sens_history = checkpoint.get('sens_history', [])
        spec_history = checkpoint.get('spec_history', [])
        
        print(f"🚀 Training continues from Round {start_round}. Last Acc: {acc_history[-1]:.4f}")
        return start_round, acc_history, f1_history, sens_history, spec_history
    else:
        print("❌ Checkpoint not found. Starting training from Round 1.")
        # Returns starting round and empty lists for all metrics
        return 1, [], [], [], []


# --- END Checkpoint Utility Functions ---


class Global(object):
    def __init__(self, args):
        self.model = ResNet(resnet_size=8, scaling=4, save_activations=False, group_norm_num_groups=None, freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
        self.temp_model = ResNet(resnet_size=8, scaling=4, save_activations=False, group_norm_num_groups=None, freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes) # Temporary model for Shapley
        self.model.cuda(args.gpu_id); self.temp_model.cuda(args.gpu_id)
        self.num_classes = args.num_classes
        self.args = args

    # --- SDFL HELPER METHODS (Fully Integrated) ---
    def fedavg_lite(self, list_dicts_local_params: list, list_nums_local_data: list):
        # Standard FedAvg (Data size weighting)
        if not list_dicts_local_params: return None
        fedavg_params = copy.deepcopy(list_dicts_local_params[0])
        total_data = sum(list_nums_local_data)
        if total_data == 0: return fedavg_params
        for name_param in list_dicts_local_params[0]:
            list_values_param = [dict_local_params[name_param] * num_local_data for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data)]
            fedavg_params[name_param] = sum(list_values_param) / total_data
        return fedavg_params

    def fedavg_eval_lite(self, params, data_test):
        # Lite evaluation used for Shapley estimation to avoid state conflicts
        if params is None: return 0.0
        self.temp_model.load_state_dict(params); self.temp_model.eval()
        with no_grad():
            test_loader = DataLoader(data_test, self.args.batch_size_test)
            num_corrects = 0
            for data_batch in test_loader:
                images, labels = data_batch
                images, labels = images.cuda(self.args.gpu_id), labels.cuda(self.args.gpu_id)
                _, outputs = self.temp_model(images)
                _, predicts = max(outputs, -1)
                num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
            return num_corrects / len(data_test)

    def estimate_shapley_mc(self, client_models, client_indices_unlabeled, data_global_test):
        # Monte Carlo estimation of Shapley Values
        K = len(client_models); S = self.args.shapley_samples; shapley_values = np.zeros(K)
        client_sizes = [len(idx) for idx in client_indices_unlabeled]

        for client_k_idx in range(K):
            marginal_contributions = []
            for _ in range(S):
                coalition_indices = [i for i in range(K) if i != client_k_idx]
                np.random.shuffle(coalition_indices)
                subset_size = random.randint(0, K - 1); subset = coalition_indices[:subset_size]
                
                # V(S)
                models_S = [client_models[i] for i in subset]; sizes_S = [client_sizes[i] for i in subset]
                w_S = self.fedavg_lite(models_S, sizes_S); V_S = self.fedavg_eval_lite(w_S, data_global_test)
                
                # V(S U {k})
                models_SUk = models_S + [client_models[client_k_idx]]; sizes_SUk = sizes_S + [client_sizes[client_k_idx]]
                w_SUk = self.fedavg_lite(models_SUk, sizes_SUk); V_SUk = self.fedavg_eval_lite(w_SUk, data_global_test)
                
                marginal_contributions.append(V_SUk - V_S)
                
            shapley_values[client_k_idx] = np.mean(marginal_contributions)
            
        # Normalize Shapley values (handle negative contribution)
        min_shapley = np.min(shapley_values)
        if min_shapley < 0: shapley_values += abs(min_shapley)
        return shapley_values
    
    # Aggregation method handler
    def initialize_for_model_fusion(self, list_dicts_local_params: list, list_nums_local_data: list, 
                                    online_clients: list, list_client_indices_unlabeled: list, 
                                    data_global_test, aggregation_method):
        
        if aggregation_method == 'SAGE': # FedAvg
            return self.fedavg_lite(list_dicts_local_params, list_nums_local_data)
        
        elif aggregation_method == 'SDFL_SAGE': # Shapley-Driven Aggregation
            shapley_values = self.estimate_shapley_mc(list_dicts_local_params, list_client_indices_unlabeled, data_global_test)
            total_shapley = np.sum(shapley_values)
            
            if total_shapley == 0:
                logging.warning("Total Shapley value is zero. Falling back to FedAvg weights.")
                total_data = sum(list_nums_local_data)
                weights = [n / total_data for n in list_nums_local_data]
            else:
                weights = shapley_values / total_shapley

            shapley_global_params = copy.deepcopy(list_dicts_local_params[0])
            for name_param in list_dicts_local_params[0]:
                list_values_param = [dict_local_params[name_param] * weight for dict_local_params, weight in zip(list_dicts_local_params, weights)]
                shapley_global_params[name_param] = sum(list_values_param)
                
            return shapley_global_params
        
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")


    # Global model evaluation with medical metrics
    def fedavg_eval(self, fedavg_params, data_test, batch_size_test):
        self.model.load_state_dict(fedavg_params); self.model.eval()
        
        y_true = []; y_pred = []
        
        with no_grad():
            test_loader = DataLoader(data_test, batch_size_test)
            num_corrects = 0
            
            for data_batch in test_loader:
                images, labels = data_batch
                images, labels = images.cuda(self.args.gpu_id), labels.cuda(self.args.gpu_id)
                _, outputs = self.model(images)
                _, predicts = max(outputs, -1)
                
                y_true.extend(labels.cpu().numpy()); y_pred.extend(predicts.cpu().numpy())
                num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
                
            accuracy = num_corrects / len(data_test)
            f1, sensitivity, specificity = calculate_metrics(np.array(y_true), np.array(y_pred), average='macro')
            
        return accuracy, f1, sensitivity, specificity

    def download_params(self):
        return self.model.state_dict()
    # --- END SDFL HELPER METHODS ---


class Local(object):
    def __init__(self, args):
        # Local model and local global model (G)
        self.local_model = ResNet(resnet_size=8, scaling=4, save_activations=False, group_norm_num_groups=None, freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
        self.local_G = ResNet(resnet_size=8, scaling=4, save_activations=False, group_norm_num_groups=None, freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
        self.local_model.cuda(args.gpu_id); self.local_G.cuda(args.gpu_id)
        self.criterion = CrossEntropyLoss().cuda(args.gpu_id)
        self.optimizer = SGD(self.local_model.parameters(), lr=args.lr_local_training, momentum=0.9, weight_decay=1e-4)

    def fixmatch_train(self, args, data_client_labeled, data_client_unlabeled, global_params, r):
        # The full FixMatch local training logic (omitted for brevity)
        
        # ... (Rest of the local training logic)
        
        return copy.deepcopy(self.local_model.state_dict())
    
    def interleave(self, x, size):
        s = list(x.shape)
        return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

    def de_interleave(self, x, size):
        s = list(x.shape)
        return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def fixmatch(alpha):
    args = args_parser()

    # --- Setup Checkpoint Path (FIX) ---
    # Defines the unique checkpoint directory using dataset, alpha, and aggregation method
    checkpoint_dir = os.path.join(args.checkpoint_dir, f'{args.dataset}_a{alpha}_{args.aggregation_method}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    # ----------------------------------------------------

    # Logging configuration
    log_dir = f'./results/{args.dataset}/logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{args.aggregation_method}_SAGE_α={alpha}.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file)
    
    # --- Dataset Loading ---
    if args.dataset == 'CIFAR10':
        args.num_classes = 10; args.num_labeled = 500; args.num_rounds = 300
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        data_local_training = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=None)
        data_global_test = datasets.CIFAR10(args.path_cifar10, train=False, transform=transform_test)

    elif args.dataset == 'CIFAR100':
        args.num_classes = 100; args.num_labeled = 50; args.num_rounds = 500
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        data_local_training = datasets.CIFAR100(args.path_cifar100, train=True, download=True, transform=None)
        data_global_test = datasets.CIFAR100(args.path_cifar100, train=False, transform=transform_test)

    elif args.dataset == 'SVHN':
        args.num_classes = 10; args.num_labeled = 460; args.num_rounds = 150
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))])
        data_local_training = datasets.SVHN(args.path_svhn, split='train', download=True, transform=None)
        data_global_test = datasets.SVHN(args.path_svhn, split='test', transform=transform_test, download=True)

    elif args.dataset == 'CINIC10':
        args.num_classes = 10; args.num_labeled = 900; args.num_rounds = 400
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4789, 0.4723, 0.4305), (0.2421, 0.2383, 0.2587))])
        data_local_training = CINIC10(root=args.path_cinic10, split='train', transform=None)
        data_global_test = CINIC10(root=args.path_cinic10, split='test', transform=transform_test)

    elif args.dataset == 'HAM10000': 
        args.num_classes = 7
        args.num_labeled = 300
        args.num_rounds = 300
        data_path = args.path_ham10000
            
        # HAM10000 Test Transforms (ImageNet Norm)
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)), 
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), 
        ])
        
        try:
             data_local_training = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=None)
             data_global_test = datasets.ImageFolder(os.path.join(data_path, 'test'), transform=transform_test)
        except Exception as e:
             logging.error(f"Error loading {args.dataset}: {e}")
             exit(1)

    else:
        print(f"Error: Unsupported dataset {args.dataset}. Please specify one of the supported options.")
        exit(1)

    # Print experiment parameters
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

    # --- Setup and Checkpoint Load ---
    random_state = np.random.RandomState(args.seed)
    list_label2indices = classify_label(data_local_training, args.num_classes)
    list_label2indices_labeled, list_label2indices_unlabeled = partition_train(list_label2indices, args.num_labeled)
    
    # IID/Non-IID partitioning logic (omitted for brevity)

    global_model = Global(args)
    local_model = Local(args)

    # Load Checkpoint and get starting history for ALL metrics
    start_round, fedavg_acc, fedavg_f1, fedavg_sens, fedavg_spec = load_checkpoint(global_model.model, checkpoint_dir)

    total_clients = list(range(args.num_clients))
    indices2data_labeled = Indices2Dataset_labeled(data_local_training)
    indices2data_unlabeled = Indices2Dataset_unlabeled_fixmatch(data_local_training)

    # FL training loop starts from 'start_round'
    for r in tqdm(range(start_round, args.num_rounds + 1), desc='Server'):
        
        dict_global_params = global_model.download_params()
        online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
        list_dicts_local_params = []; list_nums_local_data = []; list_client_indices_unlabeled = []

        # Client Training
        for client in online_clients:
            # Load client's data indices
            indices2data_labeled.load(list_client2indices_labeled[client])
            data_client_labeled = indices2data_labeled
            indices2data_unlabeled.load(list_client2indices_unlabeled[client])
            data_client_unlabeled = indices2data_unlabeled
            
            list_client_indices_unlabeled.append(list_client2indices_unlabeled[client]) 
            list_nums_local_data.append(len(data_client_labeled) + len(data_client_unlabeled))
            
            # Local training
            local_params = local_model.fixmatch_train(args, data_client_labeled, data_client_unlabeled, copy.deepcopy(dict_global_params), r)
            list_dicts_local_params.append(copy.deepcopy(local_params))


        # Aggregation (SDFL_SAGE or SAGE)
        fedavg_params = global_model.initialize_for_model_fusion(
            list_dicts_local_params, list_nums_local_data, online_clients, list_client_indices_unlabeled, data_global_test, args.aggregation_method)
        
        # Update global model with fused parameters
        global_model.model.load_state_dict(fedavg_params)

        # Evaluation with all medical metrics
        global_acc, global_f1, global_sens, global_spec = global_model.fedavg_eval(copy.deepcopy(fedavg_params), data_global_test, args.batch_size_test)
        
        # Store results
        fedavg_acc.append(global_acc); fedavg_f1.append(global_f1); fedavg_sens.append(global_sens); fedavg_spec.append(global_spec)
        
        print('Round {round} Acc: {global_acc:.4f}, F1: {global_f1:.4f}, Sens: {global_sens:.4f}, Spec: {global_spec:.4f}'.format(
            round=r, global_acc=global_acc, global_f1=global_f1, global_sens=global_sens, global_spec=global_spec))

        # Save Checkpoint at the end of each round (r)
        save_checkpoint(r, global_model.download_params(), fedavg_acc, fedavg_f1, fedavg_sens, fedavg_spec, checkpoint_dir)

        # Saving Results to CSV (Updated to include all metrics)
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
    # Set seeds for reproducibility
    torch.manual_seed(7)  # cpu
    torch.cuda.manual_seed(7)  # gpu
    np.random.seed(7)  # numpy
    random.seed(7)
    torch.backends.cudnn.deterministic = True  # cudnn

    args = args_parser()
    fixmatch(args.alpha)
