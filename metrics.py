import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'average_precision': average_precision, 'roc_auc': roc_auc}


def evaluate_link_prediction_linear(model, loss_func, node_embeddings, start_t, end_t, edges, neg_edges, device):
    # store train losses and metrics
    evaluate_losses, evaluate_metrics = [], []

    for t in range(start_t, end_t):
        if neg_edges[t].shape[1] == 0:
            continue
        embedding = node_embeddings[t - 1]
        losses, metrics = [], []
        src_node_embedding = embedding[edges[t][0]].to(device)
        dst_node_embedding = embedding[edges[t][1]].to(device)
        neg_src_node_embedding = embedding[neg_edges[t][0]].to(device)
        neg_dst_node_embedding = embedding[neg_edges[t][1]].to(device)

        with torch.no_grad():
            positive_probabilities = model(input_1=src_node_embedding,
                                           input_2=dst_node_embedding).squeeze(dim=-1).sigmoid()
            negative_probabilities = model(input_1=neg_src_node_embedding,
                                           input_2=neg_dst_node_embedding).squeeze(dim=-1).sigmoid()

            predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
            labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)],
                           dim=0)

            loss = loss_func(input=predicts, target=labels)

            losses.append(loss.item())
            metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

        evaluate_losses.append(np.mean(losses))
        evaluate_metrics.append({'average_precision': np.mean([metric['average_precision'] for metric in metrics]),
                                 'roc_auc': np.mean([metric['roc_auc'] for metric in metrics])})

    return evaluate_losses, evaluate_metrics
