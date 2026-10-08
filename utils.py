import torch
import torch.sparse
from torch_geometric.utils import to_torch_coo_tensor
import os
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB, Amazon, Coauthor, WikiCS, Airports
from torch_geometric.utils import remove_self_loops
import torch_geometric.transforms as T

DATA_ROOT = '/home/zxd/datasets/small'

# 【修改这里】：改回你原来环境中适配的导入路径
from torch_geometric.nn.conv.gcn_conv import gcn_norm

def drop_feature(x, drop_prob):
    """随机丢弃节点特征"""
    drop_mask = torch.rand(x.size(0), device=x.device) > drop_prob
    return x * drop_mask.unsqueeze(1)

def drop_edge(edge_index, drop_prob):
    """随机丢弃边"""
    num_edges = edge_index.size(1)
    drop_mask = torch.rand(num_edges, device=edge_index.device) > drop_prob
    return edge_index[:, drop_mask]

def compute_normalized_topology(edge_index, num_nodes):
    """预处理：计算对称归一化的邻接矩阵边权重"""
    edge_index_norm, edge_weight_norm = gcn_norm(
        edge_index, edge_weight=None, num_nodes=num_nodes, add_self_loops=True
    )
    return edge_index_norm, edge_weight_norm

@torch.no_grad()
def compute_ppr_matrix(edge_index, num_nodes, alpha=0.15, max_iter=10):
    """
    【创新点二】使用迭代法计算 PPR 亲和力矩阵，替代暴力的 3-hop 矩阵乘法
    """
    # 这里直接使用上面导入的 gcn_norm
    edge_index_norm, edge_weight_norm = gcn_norm(edge_index, num_nodes=num_nodes, add_self_loops=True)
    A_tilde = torch.sparse_coo_tensor(edge_index_norm, edge_weight_norm, size=(num_nodes, num_nodes)).coalesce()
    
    # 初始状态与单位阵
    I = torch.eye(num_nodes, device=edge_index.device)
    P_dense = I.clone()
    
    # 迭代近似计算 PPR
    for _ in range(max_iter):
        P_dense = (1 - alpha) * torch.sparse.mm(A_tilde, P_dense) + alpha * I
        
    return P_dense

def load_data(dataset_name, data_root=None):
    dataset_root = data_root or os.environ.get('THOT_GC_DATA_ROOT', DATA_ROOT)

    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(dataset_root, dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['computers', 'photo']:
        dataset = Amazon(dataset_root, dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['brazil', 'europe', 'usa']:
        dataset = Airports(dataset_root, dataset_name)

    return dataset