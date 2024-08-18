import argparse
import random
import time
import os
import numpy as np
from torch import nn
import wandb
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


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# Parse Args
args = get_configs()

print(args)

fix_seed(args.seed)

device = args.device

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
print('MODEL:', model)

# Loss Func
optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
loss_func = nn.BCELoss()

# Save model
if args.save_model:
    os.makedirs(f"./saved_models/", exist_ok=True)
    save_model_path = f"./saved_models/REDGSL_{args.dataset}_{str(time.time())}.pkl"
    early_stop_helper = early_stopping(patience=args.patience, save_model_path=save_model_path, model_name='REDGSL')

# Training
best_val_acc = 0.0
patience = 0
for epoch in range(args.epochs):
    print(f"*** Epoch {epoch + 1} starts ***")

    # store train losses and metrics
    train_metrics, train_losses = [], []
    val_losses, val_metrics = [], []
    test_losses, test_metrics = [], []

    # ————————Train——————————
    model.train()
    optimizer.zero_grad()

    embs, edge_losses, inter_loss = model[0](node_features, all_adjs, args.trainlength)

    for t in range(1, args.trainlength):
        embedding = embs[t - 1]
        src_node_embedding = embedding[pos_edges[t][0]].to(args.device)
        dst_node_embedding = embedding[pos_edges[t][1]].to(args.device)
        neg_src_node_embedding = embedding[neg_edges[t][0]].to(args.device)
        neg_dst_node_embedding = embedding[neg_edges[t][1]].to(args.device)

        positive_probabilities = model[1](input_1=src_node_embedding,
                                          input_2=dst_node_embedding).squeeze(dim=-1).sigmoid()
        negative_probabilities = model[1](input_1=neg_src_node_embedding,
                                          input_2=neg_dst_node_embedding).squeeze(dim=-1).sigmoid()

        predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
        labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

        loss_lp = loss_func(input=predicts, target=labels)
        intra_loss = - args.beta1 * sum(edge_losses[t - 1]) / len(edge_losses[t - 1])
        loss_pri = intra_loss + args.gamma * inter_loss[t - 1]
        total_loss = loss_lp + args.mu * loss_pri

        train_losses.append(total_loss)
        train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

    total_loss = torch.mean(torch.stack(train_losses))

    print(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {total_loss.item():.4f}')
    for metric_name in train_metrics[0].keys():
        print(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')

    total_loss.backward()
    optimizer.step()

    model.eval()
    embs, _, _ = model[0](node_features, all_adjs, timestamp)

    val_losses, val_metrics = evaluate_link_prediction_linear(model=model[1], loss_func=loss_func,
                                                              node_embeddings=embs, edges=pos_edges,
                                                              neg_edges=neg_edges, device=args.device,
                                                              start_t=val_start_t, end_t=val_end_t)

    print(f'validate loss: {np.mean(val_losses):.4f}')
    for metric_name in val_metrics[0].keys():
        print(f'validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}')

    val_acc = np.mean([val_metric.get('average_precision') for val_metric in val_metrics])

    # do test if test_epoch
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience = 0
        test_losses, test_metrics = evaluate_link_prediction_linear(model=model[1], loss_func=loss_func,
                                                                    node_embeddings=embs, edges=pos_edges,
                                                                    neg_edges=neg_edges, device=args.device,
                                                                    start_t=test_start_t, end_t=test_end_t)

        print(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            print(f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')

        best_test_mertic = []
        for metric_name in test_metrics[0].keys():
            best_test_mertic.append({metric_name: np.mean([test_metric[metric_name] for test_metric in test_metrics])})

        if args.save_model:
            early_stop_metrics = []
            for metric_name in val_metrics[0].keys():
                early_stop_metrics.append((metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), True))
            early_stop_flag = early_stop_helper.step(early_stop_metrics, model)

    else:
        patience = patience + 1
        if patience >= args.patience:
            break

print(f'get final performance on {args.dataset}')
for metric_name in best_test_mertic[0].keys():
    print(f'test {metric_name}, {best_test_mertic[0][metric_name]:.4f}')
