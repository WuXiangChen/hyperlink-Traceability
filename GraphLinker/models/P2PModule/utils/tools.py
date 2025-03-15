import pickle
from turtle import pos

import torch
import dgl
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import dgl
import numpy as np

def transfer_batch_rows(adjlist_u, edge_metapath_indices_u, art_art_batch):
    """
    Process each row in the art_art_batch and append corresponding adjacency lists and indices
    based on the mode value.

    Args:
        adjlist_u (list): List of adjacency lists for the nodes.
        edge_metapath_indices_u (list): List of edge metapath indices for the nodes.
        art_art_batch (list): The batch of art-art relations, where each element is a tuple (node1, node2, mode).

    Returns:
        tuple: Two lists - `temp_adjlist` (adjacency list) and `temp_indice` (edge indices).
    """
    temp_adjlist = [[],[]]
    temp_indice = [[],[]]
    
    # Process each row based on the mode
    for row in art_art_batch:
        mode_ = row[2]  # Extract the mode (0, 1, or other)
        # Process the pair based on the mode
        if mode_ == 0:
            temp_adjlist[0].append(adjlist_u[0][row[0]])
            temp_indice[0].append(edge_metapath_indices_u[0][row[0]])
            temp_adjlist[1].append(adjlist_u[1][row[1]])
            temp_indice[1].append(edge_metapath_indices_u[1][row[1]])
        elif mode_ == 1:
            temp_adjlist[0].append(adjlist_u[0][row[0]])
            temp_indice[0].append(edge_metapath_indices_u[0][row[0]])
            temp_adjlist[1].append(adjlist_u[0][row[1]])
            temp_indice[1].append(edge_metapath_indices_u[0][row[1]])
        else:
            temp_adjlist[0].append(adjlist_u[1][row[0]])
            temp_indice[0].append(edge_metapath_indices_u[1][row[0]])
            temp_adjlist[1].append(adjlist_u[1][row[1]])
            temp_indice[1].append(edge_metapath_indices_u[1][row[1]])

    return temp_adjlist, temp_indice

def parse_adjlist_LastFM(adjlist, edge_metapath_indices, samples=None, exclude=None, offset=None):
    edges = []
    nodes = set()
    result_indices = []
    for row, indices in zip(adjlist, edge_metapath_indices):
        row_parsed = list(map(int, row.split(' ')))
        nodes.add(row_parsed[0])
        if len(row_parsed) > 1:
            unique, counts = np.unique(row_parsed[1:], return_counts=True)
            p = []
            for count in counts:
                p += [(count ** (3 / 4)) / count] * count
            p = np.array(p)
            p = p / p.sum()
            samples = min(samples, len(row_parsed) - 1)
            sampled_idx = np.sort(np.random.choice(len(row_parsed) - 1, samples, replace=False, p=p))
            if len(row_parsed)-1!=indices.shape[0]:
                print("Er")
            neighbors = [row_parsed[i + 1] for i in sampled_idx]
            result_indices.append(indices[sampled_idx])
        for dst in neighbors:
            nodes.add(dst)
            edges.append((row_parsed[0], dst))
    mapping = {nodeId: idx for idx, nodeId in enumerate(sorted(nodes))}
    edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))
    result_indices = np.vstack(result_indices)
    return edges, result_indices, len(nodes), mapping

def get_art2art_pair(query1, target1, adjlist, nodes):
    partial_edges = []
    partial_edges.append([query1, query1])
    for adj in adjlist:
        if adj!=target1:
            nodes.add(adj)
            partial_edges.append([query1, adj])
    return partial_edges, nodes

def process_art_user_batch(temp_adjlists, temp_indices, art_art_batch, offsets, samples, 
                           device, g_lists, result_indices_lists, idx_batch_mapped_lists):
    """
    Process each row in the art_art_batch and generate a graph for each batch.
    Returns:
        None, but modifies g_lists, result_indices_lists, and idx_batch_mapped_lists.
    """
    for idx, temp_adjlist in enumerate(temp_adjlists):
        temp_indice = temp_indices[idx]
        edges, result_indices, num_nodes, mapping = parse_adjlist_LastFM(temp_adjlist, temp_indice, samples, offset=offsets)
        g = dgl.DGLGraph(multigraph=True)
        g.add_nodes(num_nodes)
        # Add edges to the graph if available
        if len(edges) > 0:
            sorted_index = sorted(range(len(edges)), key=lambda i: edges[i])
            g.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))
            result_indices = torch.LongTensor(result_indices[sorted_index]).to(device)
        else:
            result_indices = torch.LongTensor(result_indices).to(device)
        # Append the graph and result indices to the lists
        g_lists[idx].append(g)
        result_indices_lists[idx].append(result_indices)
        # Map the indices and append to the batch mapped list
        idx_batch_mapped_lists[idx].append(np.array([mapping[row[idx]] for row in art_art_batch]))
    return g_lists, result_indices_lists, idx_batch_mapped_lists

def process_art_art_batch(TrainPos_ArtAdj, art_art_batch, offsets, device,  g_lists, result_indices_lists, idx_batch_mapped_lists):
    requested_node_pair = []
    nodes_set = set()
    # Process each row based on the mode
    for row in art_art_batch:
        mode_ = row[2]  # Extract the mode (0, 1, or other)
        if mode_ == 0:
            art_1, art_2 = row[0]+offsets[0], row[1]+offsets[1]
            requested_node_pair.append([art_1, art_2])
        elif mode_ == 1:
            art_1, art_2 = row[0]+offsets[0], row[1]+offsets[0]
            requested_node_pair.append([art_1, art_2])            
        else:
            art_1, art_2 = row[0]+offsets[1], row[1]+offsets[1]
            requested_node_pair.append([art_1, art_2])           
        nodes_set.add(art_1)
        nodes_set.add(art_2)
    

    edges_T = [[],[]]
    for p2p in requested_node_pair:
        n1, n2 = p2p[0], p2p[1]
        n1_adj = TrainPos_ArtAdj[n1]
        n2_adj = TrainPos_ArtAdj[n2]
        partial_edges_n1, nodes_set = get_art2art_pair(n1, n2, n1_adj, nodes_set)
        edges_T[0].extend(partial_edges_n1)
        partial_edges_n2, nodes_set = get_art2art_pair(n2, n1, n2_adj, nodes_set)
        edges_T[1].extend(partial_edges_n2)
    

    for idx, edges_ in enumerate(edges_T):
        result_indices = np.vstack(edges_)
        mapping = {nodeId: idx for idx, nodeId in enumerate(sorted(nodes_set))}
        edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges_))
        g = dgl.DGLGraph(multigraph=True)
        g.add_nodes(len(nodes_set))
        if len(edges) > 0:
            sorted_index = sorted(range(len(edges)), key=lambda i: edges[i])
            g.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))
            result_indices = torch.LongTensor(result_indices[sorted_index]).to(device)
        else:
            raise f"Warning: edges created not existed, its' length is {len(edges)}"
        g_lists[idx].append(g)
        result_indices_lists[idx].append(result_indices)
        idx_batch_mapped_lists[idx].append(np.array([mapping[row[idx]] for row in requested_node_pair]))

    return g_lists, result_indices_lists, idx_batch_mapped_lists


def parse_minibatch_LastFM(TrainPos_ArtAdj, adjlist_u, edge_metapath_indices_u, art_art_batch, device, samples=None, offsets=None):
    # 初始化两个空列表,分别用于存储图对象和结果索引
    g_lists = [[], []]
    result_indices_lists = [[], []]
    idx_batch_mapped_lists = [[], []]
    temp_adjlist, temp_indice = transfer_batch_rows(adjlist_u, edge_metapath_indices_u, art_art_batch)
    # 遍历两种模式(一种是以user为中心的跳转模式，另一种是以已知art-art关联关系为核心的关联模式)  
    process_art_art_batch(TrainPos_ArtAdj, art_art_batch, offsets, device,  g_lists, result_indices_lists, idx_batch_mapped_lists)
    process_art_user_batch(temp_adjlist, temp_indice, art_art_batch, offsets, samples, device, g_lists, result_indices_lists, idx_batch_mapped_lists)
    return g_lists, result_indices_lists, idx_batch_mapped_lists

class index_generator:
    
    def __init__(self, batch_size, num_data=None, indices=None, shuffle=True):
        if num_data is not None:
            self.num_data = num_data
            self.indices = np.arange(num_data)
        if indices is not None:
            self.num_data = len(indices)
            self.indices = np.copy(indices)
        self.batch_size = batch_size
        self.iter_counter = 0
        self.shuffle = shuffle
        if shuffle:
            np.random.seed(42)
            np.random.shuffle(self.indices)

    def next(self):
        if self.num_iterations_left() <= 0:
            self.reset()
        self.iter_counter += 1
        return np.copy(self.indices[(self.iter_counter - 1) * self.batch_size:self.iter_counter * self.batch_size])

    def num_iterations(self):
        return int(np.ceil(self.num_data / self.batch_size))

    def num_iterations_left(self):
        return self.num_iterations() - self.iter_counter

    def reset(self):
        if self.shuffle:
            np.random.seed(42)
            np.random.shuffle(self.indices)
        self.iter_counter = 0


def read_adjlist_file(file_path):
    with open(file_path, 'r') as in_file:
        return [line.strip() for line in in_file]


def read_pickle_file(file_path):
    with open(file_path, 'rb') as in_file:
        return pickle.load(in_file)

# 注意这里的整体初始化 可能会改变已有的训练结果
import torch.nn as nn
from dgl.nn.pytorch.conv import SAGEConv
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.114)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, SAGEConv):
            nn.init.xavier_normal_(m.fc_neigh.weight, gain=1.114)  # SAGEConv的线性层初始化
            nn.init.xavier_normal_(m.fc_self.weight, gain=1.114)  # SAGEConv的线性层初始化
            if m.fc_neigh.bias is not None:
                nn.init.constant_(m.fc_neigh.bias, 0)
            if m.fc_self.bias is not None:
                nn.init.constant_(m.fc_self.bias, 0)

from sklearn.model_selection import train_test_split
def split_train_validation(train_pos, train_neg, val_size=0.2):
    """
    Splits the training dataset into a smaller training set and a validation set.
    Args:
        train_pos_issue_pr: Positive samples in the training dataset (numpy array).
        train_neg_issue_pr: Negative samples in the training dataset (numpy array).
        val_size: Proportion of the training dataset to use for validation (default: 0.2).
    Returns:
        A dictionary containing the splits for train and validation sets.
    """
    # Split positive samples
    train_pos, val_pos = train_test_split(train_pos, test_size=val_size, random_state=42)
    
    # Split negative samples
    train_neg, val_neg = train_test_split(train_neg, test_size=val_size, random_state=42)
    
    return np.array(train_pos), np.array(train_neg), np.array(val_pos), np.array(val_neg)
