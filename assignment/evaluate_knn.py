'''
Revised from squareRoot3/Rethinking-Anomaly-Detection
Almost the same with exploration1, but use knn to construct adj_mat
https://github.com/squareRoot3/Rethinking-Anomaly-Detection
Author: Zhuoyang CHEN
'''

import dgl
import torch
import numpy as np
import random
import argparse
import time
from dataset import Dataset
from utils import *
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.metrics.pairwise import cosine_similarity
import networkx
from scipy import sparse
from torch_geometric.utils.convert import from_scipy_sparse_matrix

def KNN_graph(g, args, sim_mat=None): # unregular
    if sim_mat == None:
        sim_mat = cosine_similarity(g.ndata['feature'])
    else:
        sim_mat = load_object(args.simMat)
    
    np.fill_diagonal(sim_mat, 0)
    adj = networkx.Graph()

    for i in range(sim_mat.shape[0]):
        k_idx = sim_mat[i].argsort()[::-1][:args.k]
        k_idx = k_idx[sim_mat[i][k_idx]>args.t]
        for j in k_idx:
            adj.add_edge(i,j)

    adj = networkx.to_scipy_sparse_matrix(adj).tocoo()
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)    

    return edge_index
    
def train(model, g, args):
    features = g.ndata['feature']
    labels = g.ndata['label']
    edges = KNN_graph(g, args)
    index = list(range(len(labels)))

    idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
                                                            train_size=args.train_ratio,
                                                            random_state=123, shuffle=True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                            test_size=0.67,
                                                            random_state=123, shuffle=True)
    train_mask = torch.zeros([len(labels)]).bool()
    val_mask = torch.zeros([len(labels)]).bool()
    test_mask = torch.zeros([len(labels)]).bool()

    train_mask[idx_train] = 1
    val_mask[idx_valid] = 1
    test_mask[idx_test] = 1
    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0., 0.

    weight = (1-labels[train_mask]).sum().item() / labels[train_mask].sum().item()

    train_mask = train_mask.to(device)
    features = features.to(device)
    edges = edges.to(device)
    epoch_time = []

    print('cross entropy weight: ', weight)
    time_start = time.time()
    for e in range(args.epoch):
        t0 = time.time()
        model.train()
        logits = model(features, edges, args.dropout)
        labels = labels.to(device)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=torch.tensor([1., weight]).cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        probs = logits.softmax(1)
        probs = probs.detach().to('cpu')
        labels = labels.cpu()
        f1, thres = get_best_f1(labels[val_mask], probs[val_mask])
        preds = np.zeros_like(labels)
        preds[probs[:, 1] > thres] = 1
        trec = recall_score(labels[test_mask], preds[test_mask])
        tpre = precision_score(labels[test_mask], preds[test_mask])
        tmf1 = f1_score(labels[test_mask], preds[test_mask], average='macro')
        tauc = roc_auc_score(labels[test_mask], probs[test_mask][:, 1].detach().numpy())
        time_per_epoch = time.time() - t0
        epoch_time.append(time_per_epoch)
        
        if best_f1 < f1:
            best_f1 = f1
            final_trec = trec
            final_tpre = tpre
            final_tmf1 = tmf1
            final_tauc = tauc
        print('Epoch {}, loss: {:.4f}, val mf1: {:.4f}, (best {:.4f})'.format(e, loss, f1, best_f1))
        

    time_end = time.time()
    print('time cost: ', time_end - time_start, 's')
    print('Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f}'.format(final_trec*100,
                                                                     final_tpre*100, final_tmf1*100, final_tauc*100))
    stats = []
    stats.append(
        {
            'k': args.k, #k-NN
            't': args.t, #threshold
            'EPOCH': args.epoch,
            'avg_time_per_epoch': sum(epoch_time) / len(epoch_time),
            'REC': final_trec*100,
            'PRE': final_tpre*100,
            'MF1': final_tmf1*100,
            'AUC': final_tauc*100,
        }
    )

    import pandas as pd
    df_stats = pd.DataFrame(data=stats)
    filename = args.output + '/' + 'stat_' + 'k' + str(args.k) + '_t' + str(args.t) + '_Epoch' + str(args.epoch) + '.csv'
    df_stats.to_csv(filename, index=False)

    return final_tmf1, final_tauc


# threshold adjusting for best macro f1
def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GNN')
    parser.add_argument("--dataset", type=str, default="tfinance",
                        help="Dataset for this model (yelp/amazon/tfinance/tsocial)")
    parser.add_argument("--train_ratio", type=float, default=0.4, help="Training ratio")
    parser.add_argument("--simMat", type=str, default=None, help="Pre-computed similarity matrix")
    parser.add_argument("--k", type=int, default=3, help="k for KNN neighbor number")
    parser.add_argument("--t", type=float, default=0.5, help="Threshold for node similarity")
    parser.add_argument("--hid_dim", type=int, default=16, help="Hidden layer dimension")
    parser.add_argument("--n_layer", type=int, default=3, help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout after convolution")
    parser.add_argument("--epoch", type=int, default=100, help="The max number of epochs")
    parser.add_argument("--run", type=int, default=1, help="Running times")
    parser.add_argument("--seed", type=int, default=123, help="Random states")
    parser.add_argument("--output", type=str, default='./results2', help="Output best metrics")

    args = parser.parse_args()
    #print(args)

    import os
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    dataset_name = args.dataset
    hidden_channel = args.hid_dim
    graph = Dataset(dataset_name).graph
    n_feature = graph.ndata['feature'].shape[1]
    num_classes = 2
    
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print(f'We will use 1 GPU')

    def set_rand_seed(seed):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    set_rand_seed(args.seed)
    device = 'cuda:0'

    if args.run == 1:
        model = GNN(hidden_channel, n_feature, num_classes, n_layer=args.n_layer)
        model = model.to(device)
        train(model, graph, args)

    else:
        final_mf1s, final_aucs = [], []
        for tt in range(args.run):
            model = GNN(hidden_channel, n_feature, num_classes, args.n_layer)
            model.to(device)
            mf1, auc = train(model, graph, args)
            final_mf1s.append(mf1)
            final_aucs.append(auc)
        final_mf1s = np.array(final_mf1s)
        final_aucs = np.array(final_aucs)
        print('MF1-mean: {:.2f}, MF1-std: {:.2f}, AUC-mean: {:.2f}, AUC-std: {:.2f}'.format(100 * np.mean(final_mf1s),
                                                                                            100 * np.std(final_mf1s),
                                                               100 * np.mean(final_aucs), 100 * np.std(final_aucs)))
