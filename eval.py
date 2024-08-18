import random

import numpy as np
from torch import nn

from config import get_configs
from data_util import *
from REDGSL import REDGSL, MergeLayer
from early_stopping import early_stopping
from metrics import get_link_prediction_metrics, evaluate_link_prediction_linear

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def adj_mul(adj_1, adj_2, N):
    adj_1_sp = torch.sparse_coo_tensor(adj_1, torch.ones(adj_1.shape[1], dtype=torch.float).to(adj_1.device), (N, N))
    adj_2_sp = torch.sparse_coo_tensor(adj_2, torch.ones(adj_2.shape[1], dtype=torch.float).to(adj_2.device), (N, N))
    adj_j = torch.sparse.mm(adj_1_sp, adj_2_sp)
    adj_j = adj_j.coalesce().indices()
    return adj_j

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Parse Args
args = get_configs()
print(args)
fix_seed(args.seed)
args, _, _, _, _ = load_data(args)
device = args.device

# Init Model
dynamic_backbone = REDGSL(n_feats=args.nfeat, hidden_channels=args.hidden_channels,
                               node_channels=args.node_channels, beta2=args.beta2, mamba_features=args.num_nodes,
                               num_layers=args.num_layers, dropout=args.dropout, lamda_1=args.lamda_1,
                               num_heads=args.num_heads, use_bn=args.use_bn, nb_random_features=args.M,
                               use_gumbel=args.use_gumbel, use_residual=args.use_residual, use_act=args.use_act,
                               use_jk=args.use_jk,
                               nb_gumbel_sample=args.K, rb_order=args.rb_order, rb_trans=args.rb_trans, tau=args.tau)
link_predictor = MergeLayer(input_dim1=args.out_channels, input_dim2=args.out_channels, hidden_dim=args.out_channels,
                            output_dim=1)
model = nn.Sequential(dynamic_backbone, link_predictor)

model.to(device)

# Loss Func
optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
loss_func = nn.BCELoss()

if args.exp_type == 'poisoning':
    os.makedirs(f"./saved_models/", exist_ok=True)
    save_model_path = f"./saved_models/REDGSL_{args.dataset}.pkl"
    early_stop_helper = early_stopping(patience=args.patience, save_model_path=save_model_path,
                                       model_name='Mamba_NodeFormer')

    early_stop_helper.load_checkpoint(model=model)

else:
    os.makedirs(f"./saved_models/", exist_ok=True)
    save_model_path = f"./saved_models/REDGSL_{args.dataname}.pkl"
    early_stop_helper = early_stopping(patience=args.patience, save_model_path=save_model_path,
                                       model_name='Mamba_NodeFormer')

    early_stop_helper.load_checkpoint(model=model)

if args.exp_type == 'structure':
    # ——————————————————test on structure attack——————————————————

    # Load Data
    print("\nPreparing train/test data...")
    data, pos_edges, neg_edges, adj_matrices = load_attack_data(args)

    timestamp = args.length
    val_start_t = args.trainlength
    val_end_t = args.trainlength + args.vallength
    test_start_t = args.trainlength + args.vallength
    test_end_t = args.length

    print("\nData processing done")

    node_embeddings = data['x']
    node_embeddings = node_embeddings.to(args.device).clone().detach()
    node_features = [node_embeddings for _ in range(timestamp)]
    adj_matrices = [matrix.to(device) for matrix in adj_matrices]

    # Adj storage for relational bias
    all_adjs = []
    for t in range(timestamp):
        t_adjs = []
        t_adjs.append(adj_matrices[t])
        adj = adj_matrices[t]
        for i in range(args.rb_order - 1):  # edge_index of high order adjacency
            adj = adj_mul(adj, adj, args.num_nodes)
            t_adjs.append(adj)
        all_adjs.append(t_adjs)

    print(f'————————————————test on structure attack———————————————')

    model.eval()
    embs, _, _ = model[0](node_features, all_adjs, timestamp)

    test_losses, test_metrics = evaluate_link_prediction_linear(model=model[1], loss_func=loss_func,
                                                                node_embeddings=embs, edges=pos_edges,
                                                                neg_edges=neg_edges, device=args.device,
                                                                start_t=test_start_t, end_t=test_end_t)

    print(f'test loss: {np.mean(test_losses):.4f}')
    for metric_name in test_metrics[0].keys():
        print(f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')

elif args.exp_type == 'feature':
    # ————————————————test on feature attack——————————————————

    print("\nPreparing train/test data...")
    data, _, _, _ = load_attack_data(args)
    args, pos_edges, neg_edges, adj_matrices, node_embeddings = load_data(args)

    timestamp = args.length
    val_start_t = args.trainlength
    val_end_t = args.trainlength + args.vallength
    test_start_t = args.trainlength + args.vallength
    test_end_t = args.length

    adj_matrices = [matrix.to(device) for matrix in adj_matrices]

    print("\nData processing done")

    # Adj storage for relational bias
    all_adjs = []
    for t in range(timestamp):
        t_adjs = []
        t_adjs.append(adj_matrices[t])
        adj = adj_matrices[t]
        for i in range(args.rb_order - 1):  # edge_index of high order adjacency
            adj = adj_mul(adj, adj, args.num_nodes)
            t_adjs.append(adj)
        all_adjs.append(t_adjs)

    print(f'——————————————test on feature attack 0.5————————————————')

    x = data["x_noise_0.5"].to(args.device).clone().detach()
    node_features = [x for _ in range(timestamp)] if len(x.shape) <= 2 else x

    model.eval()
    embs, _, _ = model[0](node_features, all_adjs, timestamp)

    test_losses, test_metrics = evaluate_link_prediction_linear(model=model[1], loss_func=loss_func,
                                                                node_embeddings=embs, edges=pos_edges,
                                                                neg_edges=neg_edges, device=args.device,
                                                                start_t=test_start_t, end_t=test_end_t)

    print(f'test loss: {np.mean(test_losses):.4f}')
    for metric_name in test_metrics[0].keys():
        print(f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')

    print(f'——————————————test on feature attack 1.0———————————————')

    x = data["x_noise_1.0"].to(args.device).clone().detach()
    node_features = [x for _ in range(timestamp)] if len(x.shape) <= 2 else x

    model.eval()
    embs, _, _ = model[0](node_features, all_adjs, timestamp)

    test_losses, test_metrics = evaluate_link_prediction_linear(model=model[1], loss_func=loss_func,
                                                                node_embeddings=embs, edges=pos_edges,
                                                                neg_edges=neg_edges, device=args.device,
                                                                start_t=test_start_t, end_t=test_end_t)

    print(f'test loss: {np.mean(test_losses):.4f}')
    for metric_name in test_metrics[0].keys():
        print(f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')

    print(f'——————————————test on feature attack 1.5———————————————')

    x = data["x_noise_1.5"].to(args.device).clone().detach()
    node_features = [x for _ in range(timestamp)] if len(x.shape) <= 2 else x

    model.eval()
    embs, _, _ = model[0](node_features, all_adjs, timestamp)

    test_losses, test_metrics = evaluate_link_prediction_linear(model=model[1], loss_func=loss_func,
                                                                node_embeddings=embs, edges=pos_edges,
                                                                neg_edges=neg_edges, device=args.device,
                                                                start_t=test_start_t, end_t=test_end_t)

    print(f'test loss: {np.mean(test_losses):.4f}')
    for metric_name in test_metrics[0].keys():
        print(f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')

elif args.exp_type == 'evasive' or 'poisoning' or 'clean':
    # Load Data
    print("\nPreparing train/test data...")
    args, pos_edges, neg_edges, adj_matrices, node_embeddings = load_data(args)

    timestamp = args.length
    val_start_t = args.trainlength
    val_end_t = args.trainlength + args.vallength
    test_start_t = args.trainlength + args.vallength
    test_end_t = args.length

    # node_features = node_embeddings.to(device)
    if len(node_embeddings) == timestamp:
        node_features = [
            node_embeddings[t].to(args.device).clone().detach()
            for t in range(timestamp)
        ]
    else:
        node_embeddings = node_embeddings.to(args.device).clone().detach()
        node_features = [node_embeddings for _ in range(timestamp)]
    # pos_edges = [edge.to(device) for edge in pos_edges]
    adj_matrices = [matrix.to(device) for matrix in adj_matrices]

    print("\nData processing done")

    # Adj storage for relational bias
    all_adjs = []
    for t in range(timestamp):
        t_adjs = []
        t_adjs.append(adj_matrices[t])
        adj = adj_matrices[t]
        for i in range(args.rb_order - 1):  # edge_index of high order adjacency
            adj = adj_mul(adj, adj, args.num_nodes)
            t_adjs.append(adj)
        all_adjs.append(t_adjs)

    print(f'—————————————————— test on {args.dataset} ——————————————————')

    model.eval()
    embs, _, _ = model[0](node_features, all_adjs, timestamp)

    test_losses, test_metrics = evaluate_link_prediction_linear(model=model[1], loss_func=loss_func,
                                                                node_embeddings=embs, edges=pos_edges,
                                                                neg_edges=neg_edges, device=args.device,
                                                                start_t=test_start_t, end_t=test_end_t)

    print(f'test loss: {np.mean(test_losses):.4f}')
    for metric_name in test_metrics[0].keys():
        print(f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')
