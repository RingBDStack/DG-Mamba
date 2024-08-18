import argparse

import torch


def get_configs():
    parser = argparse.ArgumentParser()
    # 1. dataset, protocol
    # parser.add_argument('--method', '-m', type=str, default='nodeformer')
    parser.add_argument('--dataset', type=str, default='collab')
    parser.add_argument('--sub_dataset', type=str, default='')
    parser.add_argument('--gpu', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--eval_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    parser.add_argument('--protocol', type=str, default='semi',
                        help='protocol for cora datasets with fixed splits, semi or supervised')
    parser.add_argument('--knn_num', type=int, default=5, help='number of k for KNN graph')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--model_dir', type=str, default='../model/')
    parser.add_argument('--exp_type', type=str, default='clean', choices=['clean', 'structure', 'feature', 'evasive', 'poisoning'], help='eval type')

    # 2. hyper-parameter for model arch and training
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--node_channels', type=int, default=32)
    parser.add_argument('--out_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--test_epochs', type=int, default=20)

    # 3. hyper-parameter for nodeformer
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--M', type=int,
                        default=30, help='number of random features')
    parser.add_argument('--use_gumbel', action='store_true', help='use gumbel softmax for message passing')
    parser.add_argument('--use_residual', action='store_true', help='use residual link for each GNN layer')
    parser.add_argument('--use_bn', action='store_true', help='use layernorm')
    parser.add_argument('--use_act', action='store_true', help='use non-linearity for each layer')
    parser.add_argument('--use_jk', action='store_true', help='concat the layer-wise results in the final layer')
    parser.add_argument('--K', type=int, default=10, help='num of samples for gumbel softmax sampling')
    parser.add_argument('--tau', type=float, default=0.25, help='temperature for gumbel softmax')
    parser.add_argument('--rb_order', type=int, default=0, help='order for relational bias, 0 for not use')
    parser.add_argument('--rb_trans', type=str, default='sigmoid', choices=['sigmoid', 'identity'],
                        help='non-linearity for relational bias')
    parser.add_argument('--batch_size', type=int, default=10000)

    # 4. hyper-parameter for Mamba
    # parser.add_argument('--mamba_features', type=int, default=307, help='number of features for the Mamba')
    parser.add_argument('--mamba_K', type=int, default=3, help='K of Mamba')
    parser.add_argument('--lamda_1', type=float, default=0.5, help='lamda for mix embedding')

    parser.add_argument('--beta1', type=float, default=0.1, help='weight for edge reg loss')
    parser.add_argument('--beta2', type=float, default=0.1, help='weight for KL-divergence')
    parser.add_argument('--gamma', type=float, default=0.1, help='weight for inter loss')
    parser.add_argument('--mu', type=float, default=0.1, help='weight for pri loss')

    parser.add_argument('--load_best_config', action='store_true', help='use best configuration')

    args = parser.parse_args()
    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'

    if args.load_best_config:
        print("use best config")
        args.num_layers = 1 
        args.hidden_channels = 32
        args.num_heads = 1 
        args.rb_order = 1 
        args.rb_trans = 'sigmoid' 
        args.M = 30 
        args.K = 10 
        args.use_bn = True 
        args.use_residual = True
        args.use_gumbel = True 
        args.epochs = 2000 
        args.beta1 = 0.1
        args.patience = 1000
        if "collab" in args.dataset:
            args.lr = 0.0025
            args.dropout = 0.0
            args.weight_decay = 0.001
            args.beta2 = 50.0
            args.gamma = 0.0025
            args.lamda_1 = 0.025

        elif "yelp" in args.dataset:
            args.lr = 0.005
            args.dropout = 0.0
            args.weight_decay = 0.0005
            args.beta2 = 100.0
            args.gamma = 0.025
            args.lamda_1 = 0.05

        elif "act" in args.dataset:
            args.lr = 0.0025
            args.dropout = 0.1
            args.weight_decay = 0.0005
            args.beta2 = 100.0
            args.gamma = 0.01
            args.lamda_1 = 0.025

    return args
