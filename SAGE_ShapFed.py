# import numpy as np
# import pandas as pd
# import copy
# import torch
# import random
# import logging
# import os
# from tqdm import tqdm
# import torch.nn.functional as F
# from torch import max, eq, no_grad
# from torch.optim import SGD
# from torch.nn import CrossEntropyLoss
# from torch.utils.data import DataLoader, RandomSampler
# from itertools import combinations
# import math
# from torchvision import datasets
# from torchvision.transforms import transforms
# from options import args_parser # Assumed to have been updated with ShapFed arguments
# from Dataset.dataset import classify_label, show_clients_data_distribution, Indices2Dataset_labeled, \
#     Indices2Dataset_unlabeled_fixmatch, partition_train
# from Dataset.sample_dirichlet import clients_indices, clients_indices_homo
# from Model.resnet import ResNet # Assumed to be your ResNet class (for SAGE)
# from Dataset.CINIC10 import CINIC10 # Included for dataset handling

# # --- Utility Functions for Checkpoints and Shapley ---

# def save_checkpoint(round_num, model_state, fedavg_acc_history, checkpoint_dir, filename='checkpoint.pt'):
#     """Saves the global model state and training history."""
#     os.makedirs(checkpoint_dir, exist_ok=True)
#     state = {
#         'round': round_num,
#         'model_state_dict': model_state,
#         'fedavg_acc': fedavg_acc_history,
#     }
#     filepath = os.path.join(checkpoint_dir, filename)
#     torch.save(state, filepath)
#     print(f"\n✅ Checkpoint saved at Round {round_num} to {filepath}")

# def load_checkpoint(model, checkpoint_dir, filename='checkpoint.pt'):
#     """Loads a checkpoint if it exists and returns the starting round and accuracy history."""
#     filepath = os.path.join(checkpoint_dir, filename)
#     if os.path.exists(filepath):
#         print(f"\n⏳ Checkpoint found at {filepath}. Loading...")
#         try:
#             # Use map_location to ensure it loads correctly regardless of original save device
#             checkpoint = torch.load(filepath, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
#             model.load_state_dict(checkpoint['model_state_dict'])
#             start_round = checkpoint['round'] + 1
#             fedavg_acc = checkpoint['fedavg_acc']
#             print(f"🚀 Training continues from Round {start_round}. Last Acc: {fedavg_acc[-1]:.4f}")
#             return start_round, fedavg_acc
#         except Exception as e:
#             print(f"--> Error loading checkpoint: {e}. Starting from scratch.")
#             return 1, []
#     else:
#         print("❌ Checkpoint not found. Starting training from Round 1.")
#         return 1, []

# def compute_cssv(args, local_models_params: list, initial_global_params, mu=0.9):
#     """
#     Computes Class-wise Similarity-based Shapley Values (CSSV).
#     The similarity is computed between the final aggregated model (weights) for a subset 
#     and the weights of a client in that subset.
#     NOTE: The original ShapFed paper used *gradients* for ResNet on CIFAR10. 
#     Here, we use the model's *weights* difference/update as a proxy for simplicity, 
#     but the true implementation should use the full gradient or weight difference if possible.
#     The most direct implementation of the ShapFed paper's CSSV uses gradients, which is hard 
#     to reconcile directly with the SAGE implementation's weight difference. 
#     We will use the *weight difference* (equivalent to the local model's "full gradient" in FedAvg context) 
#     as Shapley features, similar to FedProx Shapley.
    
#     For SAGE: Since we don't have access to the exact full local gradient $g_i$, we use the difference 
#     in model weights $\Delta w_i = w_i^{local} - w^{global}$.
#     We calculate CSSV based on the similarity between a client's *model update* $\Delta w_i$ and the 
#     *aggregated model update* $w_{S \cup \{i\}} - w_S$. 
    
#     Due to the complexity and computational cost of Monte Carlo Shapley in the Fed-Avg 
#     like formulation, we will simplify the CSSV to be based on the similarity of the 
#     client's **local model updates** $\Delta w_i$ to the **FedAvg aggregated model update** to represent 'contribution' efficiently, as often done in Shapley-inspired FL papers.

#     Let's stick to the original ShapFed idea (Similarity to a fully aggregated model) 
#     but simplify the Shapley Calculation (Monte Carlo instead of All Subsets).
#     """
    
#     num_clients = len(local_models_params)
#     if num_clients == 0:
#         return np.array([]), np.array([])

#     num_classes = args.num_classes
#     # The name of the final layer's weight and bias in your ResNet model
#     # Based on Model/resnet.py: self.classifier = nn.Linear(..., out_features=self.num_classes)
#     weight_layer_name = 'classifier.weight'
#     bias_layer_name = 'classifier.bias'
    
#     # Store client model updates (difference from global model)
#     client_updates = []
#     for local_params in local_models_params:
#         update = {}
#         for name in local_params:
#             if name in initial_global_params:
#                  # Calculate the local model update: w_local - w_global 
#                 update[name] = local_params[name] - initial_global_params[name].to(local_params[name].device)
#             else:
#                 # Handle newly added layers/params if necessary (usually they are not aggregated/updated)
#                 update[name] = torch.zeros_like(local_params[name]) 
#         client_updates.append(update)

    
#     # Use Monte Carlo sampling as an efficient proxy for Shapley
#     shapley_values = np.zeros(num_clients)
#     num_samples = args.shapley_samples # e.g., 10 or 50

#     for _ in range(num_samples):
#         # 1. Select a random permutation of clients
#         permutation = np.random.permutation(num_clients)
        
#         # 2. Iterate through the permutation
#         for i, client_index in enumerate(permutation):
            
#             # The client i is the one whose contribution is being measured: client_index
#             client_update = client_updates[client_index]
            
#             # The coalition S is the clients *before* client i in the permutation
#             coalition_indices = permutation[:i]
            
#             # Coalition S + {i}
#             coalition_S_plus_i_indices = permutation[:i+1]

#             # Calculate the average update for S + {i} coalition (Aggregated Update)
#             # This is the 'marginal contribution' needed for Shapley
            
#             # Calculate the aggregated model update for the current coalition
#             # We use the FedAvg style aggregation of the *updates*
            
#             # Aggregation for S + {i}
#             if len(coalition_S_plus_i_indices) == 0:
#                 agg_update_S_plus_i = {name: torch.zeros_like(client_update[name]) for name in client_update}
#             else:
#                 agg_update_S_plus_i = {}
#                 # Simple unweighted average of updates, as client data size is already factored into FedAvg initialization
#                 # For Shapley, often unweighted is preferred unless the coalition size factor is managed.
                
#                 # For simplicity, we use the FedAvg method on UPDATES: $\frac{1}{|S|} \sum_{j \in S} \Delta w_j$
#                 # NOTE: For SAGE, we are not measuring the total utility, but the *similarity* contribution.
#                 # The total Shapley value from the original paper needs to be adapted for this context.
                
#                 # The ShapFed contribution $v(S)$ is the cosine similarity between the aggregated 
#                 # model and the final global model. Here we approximate:
                
#                 # --- Simplified ShapFed-like CSSV for SAGE ---
#                 # Contribution is measured as the cosine similarity between:
#                 # 1. The local model update of the client $i$: $\Delta w_i$
#                 # 2. The aggregated update of all clients in the *entire* coalition $S \cup \{i\}$
                
#                 # Full coalition S + {i} Update
#                 total_coef = 0
#                 for name in client_update:
#                     agg_update_S_plus_i[name] = torch.zeros_like(client_update[name])
                
#                 for idx in coalition_S_plus_i_indices:
#                     # In SAGE/FedAvg, the contribution is often proportional to the data size.
#                     # ShapFed uses *gradients* similarity, independent of data size. 
#                     # Here, we keep it simple: unweighted average of updates for the coalition.
#                     current_client_update = client_updates[idx]
#                     for name in current_client_update:
#                         if name in agg_update_S_plus_i:
#                              agg_update_S_plus_i[name] += current_client_update[name]
#                     total_coef += 1
                
#                 if total_coef > 0:
#                     for name in agg_update_S_plus_i:
#                         agg_update_S_plus_i[name] /= total_coef
                
                
#                 # Cosine Similarity for the final layer (per class)
#                 client_layer_update = client_update[weight_layer_name]
#                 coalition_layer_update = agg_update_S_plus_i[weight_layer_name]
                
#                 num_dims = client_layer_update.shape[1]
                
#                 # Flatten the final layer (Weight and Bias) and normalize (per class for CSSV)
#                 similarity_per_class = []
#                 for cls_id in range(num_classes):
                    
#                     # Client i's update for class cls_id
#                     w_i = torch.cat([
#                         client_update[weight_layer_name][cls_id].view(-1),
#                         client_update[bias_layer_name][cls_id].view(-1)
#                     ])
#                     # Coalition S + {i}'s aggregated update for class cls_id
#                     w_S_plus_i = torch.cat([
#                         agg_update_S_plus_i[weight_layer_name][cls_id].view(-1),
#                         agg_update_S_plus_i[bias_layer_name][cls_id].view(-1)
#                     ])
                    
#                     # Normalize before computing similarity
#                     w_i_norm = F.normalize(w_i.unsqueeze(0), p=2)
#                     w_S_plus_i_norm = F.normalize(w_S_plus_i.unsqueeze(0), p=2)
                    
#                     # Cosine Similarity
#                     sim = F.cosine_similarity(w_i_norm, w_S_plus_i_norm).item()
#                     similarity_per_class.append(sim)
                    
#                 # Average similarity across all classes for this client in this permutation
#                 # The actual Shapley value is complicated; this approximation is often used in practice:
#                 marginal_contribution = np.mean(similarity_per_class)
                
#                 # Apply the Shapley-like sum
#                 shapley_values[client_index] += marginal_contribution

#     # Average the sampled contributions to get the Shapley value
#     if num_samples > 0:
#         shapley_values /= num_samples
        
#     # The Shapley values can be negative. Normalize to [0, 1] for weights.
#     min_shap = np.min(shapley_values)
#     shapley_values_shifted = shapley_values - min_shap
    
#     # Final normalized weights
#     normalized_weights = shapley_values_shifted / np.sum(shapley_values_shifted) if np.sum(shapley_values_shifted) > 0 else np.ones_like(shapley_values) / num_clients

#     # We also return a non-normalized version if needed, but the core need is the weights.
#     return shapley_values, normalized_weights


# # --- Class Definitions ---

# class Global(object):
#     def __init__(self, args):
#         self.model = ResNet(resnet_size=8, scaling=4,
#                             save_activations=False, group_norm_num_groups=None,
#                             freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
#         self.model.cuda(args.gpu_id)
#         self.num_classes = args.num_classes

#     # Renamed to clarify the original FedAvg functionality.
#     def fedavg_fusion(self, list_dicts_local_params: list, list_nums_local_data: list):
#         """Standard FedAvg: Aggregation weighted by client data size."""
#         fedavg_global_params = copy.deepcopy(list_dicts_local_params[0])
#         total_data = sum(list_nums_local_data)
        
#         for name_param in list_dicts_local_params[0]:
#             list_values_param = []
#             for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
#                 # Weight by data size
#                 list_values_param.append(dict_local_params[name_param] * num_local_data)
            
#             value_global_param = sum(list_values_param) / total_data
#             fedavg_global_params[name_param] = value_global_param
        
#         return fedavg_global_params

#     def shapfed_fusion(self, args, list_dicts_local_params: list, list_nums_local_data: list, initial_global_params):
#         """Shapley-Driven Fusion: Aggregation weighted by Shapley values."""
        
#         # 1. Compute Shapley-derived weights (normalized contributions)
#         _, normalized_shapley_weights = compute_cssv(args, list_dicts_local_params, initial_global_params, mu=0.9)
        
#         if len(normalized_shapley_weights) != len(list_dicts_local_params):
#              # Fallback to FedAvg if something goes wrong with Shapley computation
#              print("⚠️ Shapley computation failed or returned incorrect size. Falling back to FedAvg.")
#              return self.fedavg_fusion(list_dicts_local_params, list_nums_local_data)

#         # 2. Perform weighted average using Shapley weights
#         shapfed_global_params = copy.deepcopy(list_dicts_local_params[0])
        
#         for name_param in list_dicts_local_params[0]:
#             list_values_param = []
#             for dict_local_params, shapley_weight in zip(list_dicts_local_params, normalized_shapley_weights):
#                 # Weight by Shapley contribution
#                 list_values_param.append(dict_local_params[name_param] * shapley_weight)
            
#             # Sum of Shapley weights is 1 (normalized), so we don't need to divide by total weight.
#             value_global_param = sum(list_values_param)
#             shapfed_global_params[name_param] = value_global_param

#         return shapfed_global_params
    
#     def fedavg_eval(self, fedavg_params, data_test, batch_size_test):
#         """Evaluate the global model."""
#         self.model.load_state_dict(fedavg_params)
#         self.model.eval()
#         with no_grad():
#             test_loader = DataLoader(data_test, batch_size_test)
#             num_corrects = 0
#             for data_batch in test_loader:
#                 images, labels = data_batch
#                 images, labels = images.cuda(args.gpu_id), labels.cuda(args.gpu_id)
#                 _, outputs = self.model(images)
#                 _, predicts = max(outputs, -1)
#                 num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
#             accuracy = num_corrects / len(data_test)
#         return accuracy

#     def download_params(self):
#         """Returns the current state dictionary of the global model."""
#         return self.model.state_dict()


# class Local(object):
#     def __init__(self, args):
#         # Local model (w_i)
#         self.local_model = ResNet(resnet_size=8, scaling=4,
#                                  save_activations=False, group_norm_num_groups=None,
#                                  freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
#         # Global model clone for SAGE pseudo-labeling (w_G)
#         self.local_G = ResNet(resnet_size=8, scaling=4,
#                               save_activations=False, group_norm_num_groups=None,
#                               freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
        
#         self.local_model.cuda(args.gpu_id)
#         self.local_G.cuda(args.gpu_id)
        
#         self.criterion = CrossEntropyLoss().cuda(args.gpu_id)
#         self.optimizer = SGD(self.local_model.parameters(), lr=args.lr_local_training, momentum=0.9, weight_decay=1e-4)

#     def fixmatch_train(self, args, data_client_labeled, data_client_unlabeled, global_params, r):
#         """SAGE local training using FixMatch with global/local ensemble for pseudo-labeling."""

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

#         # Save initial weights for Shapley calculation (model update)
#         initial_params = copy.deepcopy(self.local_model.state_dict())

#         for local_epoch in range(args.local_epochs):
#             labeled_iter = iter(self.labeled_trainloader)
#             unlabeled_iter = iter(self.unlabeled_trainloader)
#             local_iter = int(len(data_client_unlabeled) / args.batch_size_local_labeled_fixmatch)

#             for epoch in range(local_iter):
                
#                 # Load data (handling iterator exhaustion)
#                 try: inputs_x, targets_x = labeled_iter.__next__()
#                 except: labeled_iter = iter(self.labeled_trainloader); inputs_x, targets_x = labeled_iter.__next__()
#                 try: inputs_u_w, inputs_u_s, targets_u_groundtruth = unlabeled_iter.__next__()
#                 except: unlabeled_iter = iter(self.unlabeled_trainloader); inputs_u_w, inputs_u_s, targets_u_groundtruth = unlabeled_iter.__next__()

#                 inputs_x, inputs_u_w, inputs_u_s = inputs_x.cuda(args.gpu_id), inputs_u_w.cuda(args.gpu_id), inputs_u_s.cuda(args.gpu_id)
#                 batch_size = inputs_x.shape[0]
                
#                 # Interleave labeled and unlabeled data for efficient computation (FixMatch style)
#                 inputs = self.interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1).cuda(args.gpu_id)
#                 targets_x = targets_x.cuda(args.gpu_id)

#                 # Forward pass
#                 _, logits = self.local_model(inputs)
#                 logits = self.de_interleave(logits, 2 * args.mu + 1)
#                 logits_x = logits[:batch_size]
#                 logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
#                 del logits

#                 # 1. Labeled Loss (Lx)
#                 Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

#                 # 2. Unlabeled Loss (Lu) (SAGE/FixMatch with dynamic pseudo-labeling)
#                 # Global pseudo-label generation
#                 _, logits_u_w_global = self.local_G(inputs_u_w.cuda(args.gpu_id))
#                 pseudo_label_global = torch.softmax(logits_u_w_global.detach() / args.T, dim=-1)
#                 max_probs_global, targets_u_global = torch.max(pseudo_label_global, dim=-1)

#                 # Local pseudo-label generation
#                 pseudo_label_local = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
#                 max_probs_local, targets_u_local = torch.max(pseudo_label_local, dim=-1)

#                 targets_u_local_one_hot = F.one_hot(targets_u_local, args.num_classes).float()
#                 targets_u_global_one_hot = F.one_hot(targets_u_global, args.num_classes).float()

#                 # Confidence masking
#                 mask_local = max_probs_local.ge(args.threshold).float()
#                 mask_global = max_probs_global.ge(args.threshold).float()

#                 # Confidence-Driven Soft Correction (CDSC - SAGE's core idea)
#                 delta_c = torch.clamp(torch.abs(max_probs_local - max_probs_global) + 1e-6, min=1e-6, max=1.0)
#                 kappa = torch.log(torch.tensor(2.0, device=args.gpu_id)) / 0.05 
#                 lambda_dynamic = torch.clamp(torch.exp(-kappa * delta_c), min=1e-6, max=1.0)

#                 # Final pseudo-label generation
#                 final_targets_u = torch.where(
#                     mask_local.unsqueeze(1).bool(),
#                     lambda_dynamic.unsqueeze(1) * targets_u_local_one_hot + (1 - lambda_dynamic).unsqueeze(1) * targets_u_global_one_hot,
#                     targets_u_global_one_hot
#                 )

#                 mask_valid = torch.max(mask_local, mask_global)
                
#                 # KL Divergence Loss
#                 logits_u_s_probs = torch.softmax(logits_u_s, dim=-1) + 1e-10
#                 final_targets_u = final_targets_u + 1e-10
#                 Lu = (F.kl_div(logits_u_s_probs.log(), final_targets_u, reduction='none').sum(-1) * mask_valid).mean()

#                 loss = Lx + args.lambda_u * Lu
#                 logging.info(f'Round {r}, Local Epoch {local_epoch}, Batch {epoch}: Lx = {Lx.item():.4f}, Lu = {Lu.item():.4f}')

#                 # Backward pass
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
        
#         # Return final weights (W_i)
#         return copy.deepcopy(self.local_model.state_dict())

#     def interleave(self, x, size):
#         s = list(x.shape)
#         return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

#     def de_interleave(self, x, size):
#         s = list(x.shape)
#         return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


# def fixmatch(alpha):
#     args = args_parser()

#     # Define a unique checkpoint directory for this run's parameters
#     checkpoint_dir = os.path.join(args.checkpoint_dir, f'{args.dataset}_a{alpha}_{args.aggregation_method}')
#     os.makedirs(checkpoint_dir, exist_ok=True)
#     print(f"Checkpoints will be saved to: {checkpoint_dir}")
    
#     # Logging setup
#     log_dir = f'./results/{args.dataset}/logs'
#     os.makedirs(log_dir, exist_ok=True)
#     log_file = os.path.join(log_dir, f'SAGE-ShapFed_{args.aggregation_method}_α={alpha}.log')
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file)
    
    # # Dataset Configuration
    # if args.dataset == 'HAM10000':
    #     # NOTE: You need to implement HAM10000 data loading, transforms, and splits.
    #     args.num_classes = 7
    #     args.num_labeled = 1000 
    #     args.num_rounds = 400


        
#         # Placeholder for data loading - Replace with actual HAM10000 loading
#         print("⚠️ NOTE: HAM10000 data loading and preprocessing must be implemented in Dataset/dataset.py and relevant files.")
#         try:
#              # Example for placeholder only - replace with your HAM10000 data loading
#             data_local_training = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=None)
#             data_global_test = datasets.CIFAR10(args.path_cifar10, train=False, transform=transform_test)
#         except Exception as e:
#             print(f"Error loading placeholder data (CIFAR10): {e}. Please implement HAM10000 data loading.")
#             return

#     elif args.dataset == 'CIFAR10':
#         args.num_classes = 10; args.num_labeled = 500; args.num_rounds = 300
#         transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
#         data_local_training = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=None)
#         data_global_test = datasets.CIFAR10(args.path_cifar10, train=False, transform=transform_test)
#     # ... (Other dataset definitions: CIFAR100, SVHN, CINIC10) ...
#     else:
#         print(f"Error: Unsupported dataset {args.dataset}."); return

#     # Data Partitioning (using Dirichlet distribution from SAGE code)
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

#     # Model and Client Initialization
#     global_model = Global(args)
#     local_model = Local(args)
#     total_clients = list(range(args.num_clients))
#     indices2data_labeled = Indices2Dataset_labeled(data_local_training)
#     indices2data_unlabeled = Indices2Dataset_unlabeled_fixmatch(data_local_training)
    
#     # Checkpoint loading
#     start_round, fedavg_acc = load_checkpoint(global_model.model, checkpoint_dir)

#     # --- FL Training Loop ---
#     for r in tqdm(range(start_round, args.num_rounds + 1), desc='Server'):
        
#         dict_global_params = global_model.download_params()
        
#         # Client Sampling
#         online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
#         list_dicts_local_params = []
#         list_nums_local_data = []

#         # Client Training
#         for client_id in online_clients:
#             indices2data_labeled.load(list_client2indices_labeled[client_id])
#             data_client_labeled = indices2data_labeled
#             indices2data_unlabeled.load(list_client2indices_unlabeled[client_id])
#             data_client_unlabeled = indices2data_unlabeled
            
#             list_nums_local_data.append(len(data_client_labeled) + len(data_client_unlabeled))
            
#             # Local training for SAGE/FixMatch
#             local_params = local_model.fixmatch_train(args, data_client_labeled, data_client_unlabeled, copy.deepcopy(dict_global_params), r)
#             list_dicts_local_params.append(copy.deepcopy(local_params))

#         # --- Model Fusion (Aggregation) ---
#         if args.aggregation_method == 'SAGE': # FedAvg (Original SAGE)
#             fusion_params = global_model.fedavg_fusion(list_dicts_local_params, list_nums_local_data)
#             print("Aggregation Method: FedAvg (SAGE)")
        
#         elif args.aggregation_method == 'SDFL_SAGE': # ShapFed-SAGE
#             # ShapFed uses Shapley weights on model updates/weights.
#             fusion_params = global_model.shapfed_fusion(args, list_dicts_local_params, list_nums_local_data, dict_global_params)
#             print("Aggregation Method: ShapFed-SAGE (Shapley-Driven)")

#         else: # Default fallback to FedAvg
#             fusion_params = global_model.fedavg_fusion(list_dicts_local_params, list_nums_local_data)
#             print("Aggregation Method: Defaulting to FedAvg.")
        
#         # Update Global Model
#         global_model.model.load_state_dict(fusion_params)

#         # Evaluation
#         global_acc = global_model.fedavg_eval(copy.deepcopy(fusion_params), data_global_test, args.batch_size_test)
#         fedavg_acc.append(global_acc)
#         logging.info(f'Round {r} accuracy:{global_acc}')
#         print(f'round {r} accuracy:{global_acc}')

#         # Save Checkpoint
#         save_checkpoint(r, global_model.download_params(), fedavg_acc, checkpoint_dir)
        
#         # Save results to CSV (as in original SAGE code)
#         result_dir = f'./results/{args.dataset}'
#         os.makedirs(result_dir, exist_ok=True)
#         result_file = f'{result_dir}/{args.aggregation_method}_α={alpha}.csv'
#         acc_num_pseudo_label_csv_index = list(range(1, len(fedavg_acc) + 1))
#         acc_num_pseudo_label_csv_df = pd.DataFrame({'acc': fedavg_acc}, index=acc_num_pseudo_label_csv_index)
#         acc_num_pseudo_label_csv_df.to_csv(result_file, encoding='utf8')

# if __name__ == '__main__':
#     # Set Seeds for Reproducibility
#     torch.manual_seed(7)
#     torch.cuda.manual_seed(7)
#     np.random.seed(7)
#     random.seed(7)
#     torch.backends.cudnn.deterministic = True

#     args = args_parser()
#     fixmatch(args.alpha)


# SAGE_ShapFed.py

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
            best_acc = max(fedavg_acc)
            print(f"[SAGE] Resuming from Round {start_round}. Last Acc: {fedavg_acc[-1] if fedavg_acc else 0:.4f}, Best Acc: {best_acc:.4f}")
            return start_round, fedavg_acc
        except Exception as e:
            print(f"[SAGE] Error loading checkpoint: {e}. Starting from scratch.")
            return 1, []
    else:
        print("[SAGE] No checkpoint found. Starting from Round 1.")
        return 1, []

# def compute_cssv(args, local_models_params, initial_global_params):
#     """
#     ShapFed: Class-Specific Shapley Values (CSSV) Calculation.
#     measure the cosine similarity between coalition update and clietn's model updates 
#     (weights - global_weights).
#     """
#     num_clients = len(local_models_params)
#     if num_clients == 0:
#         return np.array([])
    
#     # SAGE ResNet modelinde son katman 'classifier' olarak tanımlı
#     weight_layer = 'classifier.weight'
#     bias_layer = 'classifier.bias'
    
#     # 1. Her istemcinin 'Model Update'ini (Delta W) hesapla
#     # Update = Local_Weights - Global_Weights
#     client_updates = []
#     for local_params in local_models_params:
#         update = {}
#         for name in local_params:
#             if name in initial_global_params:
#                 # GPU üzerinde işlem yap
#                 update[name] = local_params[name] - initial_global_params[name].to(local_params[name].device)
#         client_updates.append(update)

#     shapley_values = np.zeros(num_clients)
#     num_samples = args.shapley_samples # Monte Carlo örnek sayısı (Örn: 10)
    
#     # 2. Monte Carlo Shapley Hesaplaması
#     for _ in range(num_samples):
#         permutation = np.random.permutation(num_clients)
        
#         for i, client_idx in enumerate(permutation):
#             # Koalisyon: Permütasyonda bu istemciden öncekiler
#             coalition_indices = permutation[:i]
#             # Koalisyon + İstemci
#             coalition_plus_indices = permutation[:i+1]
            
#             # Koalisyonun birleşik güncellemesini hesapla (Basit Ortalaması)
#             # ShapFed mantığında gradyan/update benzerliği esastır.
            
#             # Mevcut istemcinin son katman update'i
#             curr_update_w = torch.cat([
#                 client_updates[client_idx][weight_layer].view(-1),
#                 client_updates[client_idx][bias_layer].view(-1)
#             ])
            
#             # Koalisyon + İstemci'nin ortalama update'i (Aggregated Update)
#             agg_update_w = torch.zeros_like(curr_update_w)
            
#             if len(coalition_plus_indices) > 0:
#                 temp_w = torch.zeros_like(client_updates[0][weight_layer])
#                 temp_b = torch.zeros_like(client_updates[0][bias_layer])
                
#                 for c_idx in coalition_plus_indices:
#                     temp_w += client_updates[c_idx][weight_layer]
#                     temp_b += client_updates[c_idx][bias_layer]
                
#                 # Ortalamasını al
#                 temp_w /= len(coalition_plus_indices)
#                 temp_b /= len(coalition_plus_indices)
                
#                 agg_update_w = torch.cat([temp_w.view(-1), temp_b.view(-1)])

#             # Cosine Similarity Hesapla
#             sim = F.cosine_similarity(curr_update_w.unsqueeze(0), agg_update_w.unsqueeze(0)).item()
            
#             # Shapley değerine ekle (Marjinal katkı olarak benzerliği kullanıyoruz)
#             shapley_values[client_idx] += sim

#     if num_samples > 0:
#         shapley_values /= num_samples

#     # Değerleri normalize et (Negatifleri temizle ve toplama böl)
#     shapley_values = np.maximum(shapley_values, 0) # ReLU gibi, negatif katkıyı 0 yap
#     total_shapley = np.sum(shapley_values)
    
#     if total_shapley > 0:
#         normalized_weights = shapley_values / total_shapley
#     else:
#         # Hepsi 0 ise eşit dağıt (FedAvg fallback)
#         normalized_weights = np.ones(num_clients) / num_clients
        
#     return normalized_weights


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
    
    # The names of the last layer parameters in your ResNet model
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
    

#Classess
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
        - SAGE: weighted average acc. to data numbet (FedAvg).
        - ShapFed: weighted average acc. to Shapley values.
        """
        fused_params = copy.deepcopy(list_dicts_local_params[0])
        
        # define aggregation weights
        if args.aggregation_method == 'ShapFed':
            # ShapFed: contribution based
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
        # initialize_for_model_fusion functionn updated
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
