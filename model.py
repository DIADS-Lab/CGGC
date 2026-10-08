import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import remove_self_loops, add_self_loops
from sklearn.cluster import KMeans
from torch_geometric.nn.conv.gcn_conv import gcn_norm

class THOT_GC(nn.Module):
    def __init__(self, in_dim, hid_dim, n_clusters=7, tau=None, args=None): 
        super().__init__()
        if args is None:
            raise ValueError("THOT_GC requires args from argparse")
        self.args = args
        
        # 1. 结构
        self.gcn_enc = nn.ModuleList([
            GCNConv(in_dim, hid_dim),
            GCNConv(hid_dim, hid_dim),
            GCNConv(hid_dim, hid_dim)
        ])
        self.gcn_norms = nn.ModuleList([
            nn.LayerNorm(hid_dim),
            nn.LayerNorm(hid_dim),
            nn.LayerNorm(hid_dim)
        ])
        self.dropout = nn.Dropout(args.gcn_dropout)
        self.feat_decoder = nn.Linear(hid_dim, in_dim)
        
        self.tau = args.tau if tau is None else tau
        self.n_clusters = n_clusters
        self.prototypes = nn.Parameter(torch.Tensor(n_clusters, hid_dim)) # 聚类中心作为可学习参数，初始值随机

        # self.register_buffer("prototypes", torch.randn(n_clusters, hid_dim))

        nn.init.xavier_uniform_(self.prototypes) # Xavier 初始化，保持初始分布均匀且不偏倚

        # 2. 属性
        self.raw_proj = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.LayerNorm(hid_dim),
            nn.PReLU(),
            nn.Linear(hid_dim, hid_dim)
        )
        
        # 融合门控参数：控制语义与拓扑的物理混合比例
        self.fusion_gate = nn.Parameter(torch.tensor(0.5)) 

    def forward(self, x, edge_index, return_decoupled=False):
        # 结构流(GCN)
        # h1 = self.dropout(F.relu(self.gcn_norms[0](self.gcn_enc[0](x, edge_index))))
        # z_topo = self.gcn_norms[1](self.gcn_enc[1](h1, edge_index))
        # z_topo = F.normalize(z_topo, dim=1)

        h1 = self.dropout(F.relu(self.gcn_norms[0](self.gcn_enc[0](x, edge_index))))
        # 第二层 GCN
        h2 = self.gcn_norms[1](self.gcn_enc[1](h1, edge_index))
        # 第三层 GCN
        h3 = self.gcn_norms[2](self.gcn_enc[2](h2, edge_index))
        # 融合两层的特征
        h_fused = self.args.layer_w1 * h1 + self.args.layer_w2 * h2 + self.args.layer_w3 * h3
        z_topo = F.normalize(h_fused, dim=1)

        # 属性流(MLP)
        z_attr = self.raw_proj(x)
        z_attr = F.normalize(z_attr, dim=1)

        # 动态自适应解耦融合
        alpha = torch.sigmoid(self.fusion_gate)
        z_fused = (1 - alpha) * z_topo + alpha * z_attr
        z_fused = F.normalize(z_fused, dim=1)

        # 在微调阶段返回解耦特征，用于计算正交损失；其他阶段保持原样防报错
        if return_decoupled:
            return z_fused, z_topo, z_attr
        return z_fused
    
    # 增强表征是根据当前的聚类的结果提取属性的共有属性，在重新计算的时候在计算节点间的相似度，同时结合归一化割思想，控制“类间低相似度”和“类内高相似度”，只增强当前类别的节点
    def enhance_representation(self, x, h, Q, epoch, max_epochs):
        # 1. 计算当前的动态选取比例 (从 5% 到 50%)
        ratio = self.args.enhance_ratio_start + self.args.enhance_ratio_start * (epoch / max_epochs)
        ratio = min(self.args.enhance_ratio_max, ratio) 
        
        Y = Q.argmax(dim=1) # 获取所有节点的伪标签
        conf = Q.max(dim=1)[0] # 获取所有节点的置信度
        
        enhanced_h = h.clone()
        
        for k in range(self.n_clusters):
            # 第一步：只锁定当前类别 k 的节点
            cluster_mask = (Y == k)
            cluster_nodes = cluster_mask.nonzero(as_tuple=False).view(-1) # 将布尔掩码转换为节点索引列表
            
            # 如果某个类被完全置空了，直接跳过防报错
            if len(cluster_nodes) == 0: continue
            
            # 第二步：对置信度进行降序排列
            cluster_conf = conf[cluster_nodes]
            _, sorted_idx = torch.sort(cluster_conf, descending=True)
            
            # 第三步：按当前类别的实际规模，严格计算出属于该类的 Top X 数量的核心节点，比例随 epoch 进度动态调整
            top_k_num = max(1, int(len(cluster_nodes) * ratio))
            top_nodes = cluster_nodes[sorted_idx[:top_k_num]]
            
            # 第四步：提取这几个核心节点的原始属性 x，算出纯净中心，再映射成 core_emb
            raw_features_core = x[top_nodes].mean(dim=0, keepdim=True)
            core_emb = F.normalize(self.raw_proj(raw_features_core), dim=1)
            
            # 第五步：只拉扯当前类别的节点，绝不干扰其他类别
            alpha = torch.sigmoid(self.fusion_gate) * self.args.enhance_alpha_scale
            enhanced_h[cluster_nodes] = (1 - alpha) * enhanced_h[cluster_nodes] + alpha * core_emb
            
        return F.normalize(enhanced_h, dim=1)

    def pretrain_loss(self, z, x_target, edge_index_target):
        N = z.size(0)
        x_rec = self.feat_decoder(z)
        loss_feat = (1.0 - F.cosine_similarity(x_rec, x_target, dim=-1)).mean()
        
        adj_logits = torch.mm(z, z.t()) / self.args.adj_temp # 温度参数，控制边权的分布范围，防止过于尖锐或过于平坦
        adj_label = torch.zeros((N, N), device=z.device) # 构建邻接矩阵标签，边存在为1，不存在为0
        adj_label[edge_index_target[0], edge_index_target[1]] = 1.0
        adj_label.fill_diagonal_(1.0)
        
        pos_weight = float(N * N - adj_label.sum()) / adj_label.sum()
        loss_topo = F.binary_cross_entropy_with_logits(
            adj_logits, adj_label, pos_weight=torch.tensor([pos_weight], device=z.device)
        )
        return self.args.feat_loss_weight * loss_feat + loss_topo

    def purify_topology(self, edge_index, Q, confidence_thresh=0.75): # 将低置信度的拓扑边过滤掉
        row, col = edge_index
        max_q, preds = Q.max(dim=1)
        
        is_intra = preds[row] == preds[col]
        is_confident = (max_q[row] > confidence_thresh) & (max_q[col] > confidence_thresh)
        
        keep_mask = ~(is_confident & (~is_intra))
        return edge_index[:, keep_mask] # 返回保留的边索引

    def get_smoothed_z(self, z, edge_index):
        N = z.size(0)
        edge_index_no_loop, _ = remove_self_loops(edge_index)
        row, col = edge_index_no_loop
        
        # 这部分直接使用余弦相似度计算是不是可以优化掉？毕竟每次都要计算全图的相似度，感觉有点重。可以考虑只计算边上的相似度，或者直接用边权来代替相似度。
        edge_weight = F.cosine_similarity(z[row], z[col], dim=-1) 
        edge_weight = F.relu(edge_weight) + 1e-4 
        
        pruned_edge_index, pruned_edge_weight = add_self_loops(
            edge_index_no_loop, edge_weight, fill_value=1.0, num_nodes=N)
        
        row_p, col_p = pruned_edge_index
        deg = torch.zeros(N, device=z.device)
        deg.scatter_add_(0, row_p, pruned_edge_weight)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        norm_weight = deg_inv_sqrt[row_p] * pruned_edge_weight * deg_inv_sqrt[col_p]
        
        adj_adaptive = torch.sparse_coo_tensor(pruned_edge_index, norm_weight, size=(N, N)).coalesce()
        
        # ==========================================
        # 突破点 1：APPNP 传送机制
        # ==========================================
        alpha = self.args.alpha # 保留自身特征的 teleport 比例
        z_smooth_1 = torch.sparse.mm(adj_adaptive, z)
        z_smooth_1 = (1 - alpha) * z_smooth_1 + alpha * z
        
        z_smooth_2 = torch.sparse.mm(adj_adaptive, z_smooth_1)
        z_smooth_2 = (1 - alpha) * z_smooth_2 + alpha * z
        
        return F.normalize(z_smooth_2, dim=1)

    def target_distribution(self, Q):
        weight = (Q ** 2) / Q.sum(0)
        return (weight.t() / weight.sum(1)).t()
    
    # 2. 提取高置信度的语义近邻边
    def get_semantic_edges(self, z, k=None):
        if k is None:
            k = self.args.k
        with torch.no_grad():
            # 计算节点间的全连接余弦相似度
            sim = torch.mm(z, z.t())
            # 排除自身连边
            sim.fill_diagonal_(-1)
            # 找到每个节点最相似的 Top-K 邻居
            _, topk_indices = torch.topk(sim, k, dim=1)
            
            row = torch.arange(z.size(0), device=z.device).repeat_interleave(k)
            col = topk_indices.view(-1)
            return torch.stack([row, col], dim=0)
        
    def filter_noise_edges(self, z, edge_index):
        row, col = edge_index
        sim = F.cosine_similarity(z[row], z[col], dim=-1)
        keep_mask = sim > self.args.noise_sim_thresh 
        return edge_index[:, keep_mask]

    @torch.no_grad()
    def init_prototypes(self, z, edge_index):
        z_smooth = self.get_smoothed_z(z, edge_index)
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.args.kmeans_n_init,
            random_state=self.args.kmeans_seed,
        )
        kmeans.fit(z_smooth.cpu().numpy())
        self.prototypes.data = F.normalize(torch.tensor(kmeans.cluster_centers_, device=z.device), dim=1) # 将聚类中心转换为张量并归一化后赋值给模型的 prototypes 参数，作为初始聚类中心

    def update_prototypes(self, z, Q):
        with torch.no_grad():
            cluster_weights = Q.sum(dim=0, keepdim=True).t() + 1e-8 # 计算每个簇的权重，防止除零错误
            empirical_centers = torch.mm(Q.t(), z) / cluster_weights # 计算每个簇的经验中心，作为当前簇的代表性特征
            momentum = self.args.proto_momentum
            self.prototypes.data = F.normalize(momentum * self.prototypes.data + (1.0 - momentum) * empirical_centers, dim=1)

    def finetune_loss(self, z_raw, edge_index, current_epoch, total_epochs):
        c = F.normalize(self.prototypes, dim=1) 
        logits_raw = torch.mm(z_raw, c.t()) / self.tau 
        Q_raw = F.softmax(logits_raw, dim=1)

        progress = current_epoch / total_epochs
        
        # 1. 拓扑纯化与语义加边 (保留高效的软过滤)
        purified_edge_index = self.filter_noise_edges(z_raw, edge_index)
        semantic_edge_index = self.get_semantic_edges(z_raw, k=self.args.k)
        combined_edge_index = torch.cat([purified_edge_index, semantic_edge_index], dim=1)

        # 2. 高效 APPNP 平滑
        z_smooth = self.get_smoothed_z(z_raw, combined_edge_index)
        logits_smooth = torch.mm(z_smooth, c.t()) / self.tau
        Q_smooth = F.softmax(logits_smooth, dim=1)

        # 3. 恢复 DEC 伪标签与 Lazy Update
        with torch.no_grad():
            if current_epoch % self.args.proto_update_interval == 1 or not hasattr(self, 'cached_P'):
                # 用平滑后的高质量 Q 去生成目标 P，自然放大长尾特征
                P = self.target_distribution(Q_smooth).detach()
                self.cached_P = P
                
                # 聚类中心的物理漂移依然保留，它能稳定聚类空间
                cluster_weights = P.sum(dim=0, keepdim=True).t() + 1e-8 
                empirical_centers = torch.mm(P.t(), z_smooth) / cluster_weights 
                momentum = self.args.proto_momentum
                self.prototypes.data = F.normalize(momentum * self.prototypes.data + (1.0 - momentum) * empirical_centers, dim=1)
            else:
                P = self.cached_P

        # 4. 恢复 KL 散度损失
        loss_kl = F.kl_div(Q_raw.log(), P, reduction='batchmean')
        
        # 5. 恢复簇级正交排斥力
        Q_t = Q_raw.t() 
        Q_t_norm = F.normalize(Q_t, p=2, dim=1)
        cluster_corr = torch.mm(Q_t_norm, Q_t_norm.t()) 
        cluster_corr.fill_diagonal_(0)
        loss_dispersion = cluster_corr.sum() / (self.n_clusters * (self.n_clusters - 1))
        
        # 随着训练推进，逐渐增加排斥力，防止后期坍塌
        dispersion_weight = min(self.args.disp_weight_max, self.args.disp_weight_scale * progress)
        
        # 终极联合损失
        loss_st = loss_kl + dispersion_weight * loss_dispersion 
        
        return loss_st, Q_raw

    @torch.no_grad()
    def get_cluster_assignments(self, z, edge_index):
        c = F.normalize(self.prototypes, dim=1)
        Q_raw = F.softmax(torch.mm(z, c.t()) / self.tau, dim=1)
        purified_edge_index = self.purify_topology(edge_index, Q_raw, confidence_thresh=self.args.confidence_thresh)
        
        z_smooth = self.get_smoothed_z(z, purified_edge_index)
        logits = torch.mm(z_smooth, c.t())
        return logits.argmax(dim=1).cpu().numpy()