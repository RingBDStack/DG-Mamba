import os

import torch
from torch_geometric.utils import remove_self_loops, add_self_loops


def load_data(args):
    dataset = args.dataset
    adj_matrices = []
    node_embeddings = []
    p_edges = []
    n_edges = []

    CUR_DIR = os.path.dirname(os.path.abspath(__file__))

    if "collab" in dataset:
        args.dataname = "collab"
        args.dataset = dataset
        args.testlength = 5
        args.vallength = 1
        args.trainlength = 10
        args.length = 16
        args.split = 0

        if "evasive" in dataset:
            print("load evasive")
            dataroot = os.path.join(CUR_DIR, 'data/evasive/')
            filename = f"{dataroot}" + "collab_evasive_" + dataset.split("_")[2]
            data = torch.load(filename)
            args.nfeat = data["x"][0].shape[1]
            args.num_nodes = len(data["x"][0])
        elif "poison" in dataset:
            print("load poison")
            dataroot = os.path.join(CUR_DIR, 'data/poisoning/')
            filename = f"{dataroot}" + "collab_poison_" + dataset.split("_")[2]
            print(filename)
            data = torch.load(filename)
            args.nfeat = data["x"][0].shape[1]
            args.num_nodes = len(data["x"][0])
        else:
            dataroot = os.path.join(CUR_DIR, "data/origin")
            processed_datafile = f"{dataroot}/collab"
            data = torch.load(f'{processed_datafile}')
            args.nfeat = data['x'].shape[1]
            args.num_nodes = len(data['x'])

        p_edges = data['train']['pedges']
        n_edges = data['train']['nedges']
        adj_matrices = get_matrix(args.num_nodes, data['train']['edge_index_list'])
        node_embeddings = data['x']

    elif "yelp" in dataset:
        args.dataname = "yelp"
        args.dataset = dataset
        args.testlength = 8
        args.vallength = 1
        args.trainlength = 15
        args.length = 24
        args.shift = 3972
        args.num_nodes = 13095
        args.split = 0

        if "evasive" in dataset:
            print("load evasive")
            dataroot = os.path.join(CUR_DIR, 'data/evasive/')
            filename = f"{dataroot}" + "yelp_evasive_" + dataset.split("_")[2]
            data = torch.load(filename)
            args.nfeat = data["x"][0].shape[1]
        elif "poison" in dataset:
            print("load poison")
            dataroot = os.path.join(CUR_DIR, 'data/poisoning/')
            filename = f"{dataroot}" + "yelp_poison_" + dataset.split("_")[2]
            print(filename)
            data = torch.load(filename)
            args.nfeat = data["x"][0].shape[1]
        else:
            dataroot = os.path.join(CUR_DIR, "data/origin")
            processed_datafile = f"{dataroot}/yelp"
            data = torch.load(f'{processed_datafile}')

            args.nfeat = data['x'].shape[1]
            args.num_nodes = len(data['x'])

        p_edges = data['train']['pedges']
        n_edges = data['train']['nedges']
        adj_matrices = get_matrix(args.num_nodes, data['train']['edge_index_list'])
        node_embeddings = data['x']

    elif "act" in dataset:
        args.dataname = "act"
        args.dataset = dataset
        args.testlength = 8
        args.vallength = 2
        args.trainlength = 20
        args.length = 30

        if "evasive" in dataset:
            
            dataroot = os.path.join(CUR_DIR, 'data/evasive/')
            filename = f"{dataroot}" + "act_evasive_" + dataset.split("_")[2]
            data = torch.load(filename)
            args.nfeat = data["x"][0].shape[1]
            args.num_nodes = len(data["x"][0])
        elif "poison" in dataset:
            print("load poison")
            dataroot = os.path.join(CUR_DIR, 'data/poisoning/')
            filename = f"{dataroot}" + "act_poison_" + dataset.split("_")[2]
            data = torch.load(filename)
            print(filename)
            args.nfeat = data["x"][0].shape[1]
            args.num_nodes = len(data["x"][0])
        else:
            dataroot = os.path.join(CUR_DIR, "data/origin")
            processed_datafile = f"{dataroot}/act"
            data = torch.load(f"{processed_datafile}")
            args.nfeat = data["x"].shape[1]
            args.num_nodes = len(data["x"])

        p_edges = data['train']['pedges']
        n_edges = data['train']['nedges']
        adj_matrices = get_matrix(args.num_nodes, data['train']['edge_index_list'])
        node_embeddings = data['x']

    return args, p_edges, n_edges, adj_matrices, node_embeddings

def load_attack_data(args):
    dataset = args.dataset
    adj_matrices = []
    node_embeddings = []
    p_edges = []
    n_edges = []

    CUR_DIR = os.path.dirname(os.path.abspath(__file__))

    if "collab" in dataset:
        args.dataname = "collab"
        args.dataset = dataset
        args.testlength = 5
        args.vallength = 1
        args.trainlength = 10
        args.length = 16
        args.split = 0

        dataroot = os.path.join(CUR_DIR, "data/origin")
        processed_datafile = f"{dataroot}/collab"
        data = torch.load(f'{processed_datafile}')
        args.nfeat = data['x'].shape[1]
        args.num_nodes = len(data['x'])

    elif "yelp" in dataset:
        args.dataname = "yelp"
        args.dataset = dataset
        args.testlength = 8
        args.vallength = 1
        args.trainlength = 15
        args.length = 24
        args.shift = 3972
        args.num_nodes = 13095
        args.split = 0

        dataroot = os.path.join(CUR_DIR, "data/origin")
        processed_datafile = f"{dataroot}/yelp"
        data = torch.load(f'{processed_datafile}')
        args.nfeat = data['x'].shape[1]
        args.num_nodes = len(data['x'])

    elif "act" in dataset:
        args.dataname = "act"
        args.dataset = dataset
        args.testlength = 8
        args.vallength = 2
        args.trainlength = 20
        args.length = 30

        dataroot = os.path.join(CUR_DIR, "data/origin")
        processed_datafile = f"{dataroot}/act"
        data = torch.load(f"{processed_datafile}")
        args.nfeat = data["x"].shape[1]
        args.num_nodes = len(data["x"])

    p_edges, n_edges = [], []
    for t in range(args.length):
        t_pos = torch.cat((data['train']['pedges'][t], data['test']['pedges'][t]), dim=1)
        t_neg = torch.cat((data['train']['nedges'][t], data['test']['nedges'][t]), dim=1)
        p_edges.append(t_pos)
        n_edges.append(t_neg)

    adj_matrices = get_matrix(args.num_nodes, p_edges)

    return data, p_edges, n_edges, adj_matrices


def get_matrix(num_nodes, edges):
    sparse_matrices = []
    for i, adj in enumerate(edges):
        adj, _ = remove_self_loops(adj)
        adj, _ = add_self_loops(adj, num_nodes=num_nodes)
        sparse_matrices.append(adj)

    return sparse_matrices

