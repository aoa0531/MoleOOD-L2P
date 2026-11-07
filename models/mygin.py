import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.utils import remove_self_loops, add_self_loops
__all__ = ['MyGIN']

# =====================================================================
# 1. 基础构件：MLP (保持原代码逻辑，使用 PyTorch 原生实现)
# =====================================================================
class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.linear_or_not = (num_layers == 1)

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linears = nn.ModuleList()
            self.batch_norms = nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for _ in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


# =====================================================================
# 2. 核心构件：自定义的 GIN 卷积层 (适配原代码的边特征处理方式)
# =====================================================================
class MyGINEConv(MessagePassing):
    """
    自定义的 GIN 卷积层。
    完全等价于原代码 DGL 中的 `fn.u_add_e` 和 `fn.sum` 操作。
    """
    def __init__(self, mlp, **kwargs):
        # aggr='add' 等价于 DGL 中的 fn.sum
        super(MyGINEConv, self).__init__(aggr='add', **kwargs)
        self.nn = mlp

    def forward(self, x, edge_index, edge_attr):
        # 统一行为：先移除所有可能存在的自环，得到一个纯净的化学键索引
        edge_index_no_loop, _ = remove_self_loops(edge_index)

        # 接下来，我们需要准备与这个纯净索引对应的边属性
        edge_attr_no_loop = None
        if edge_attr is not None:
            # 【解决 attr > index】
            # 无论 edge_attr 原来多长，都只取与真实化学键数量对应的部分
            num_real_edges = edge_index_no_loop.size(1)
            if edge_attr.size(0) >= num_real_edges:
                edge_attr_no_loop = edge_attr[:num_real_edges]
            else:
                # 如果属性比真实边还少（异常情况），则用 0 补齐
                padding = torch.zeros((num_real_edges - edge_attr.size(0), edge_attr.size(1)), device=edge_attr.device)
                edge_attr_no_loop = torch.cat([edge_attr, padding], dim=0)
        
        # 统一加上新的自环，得到最终的边索引
        edge_index_with_loop, _ = add_self_loops(edge_index_no_loop, num_nodes=x.size(0))

        # 为新增的自环准备属性
        final_edge_attr = None
        if edge_attr_no_loop is not None:
            # 【解决 index > attr】
            # 创建数量等于节点数的“虚拟”属性（全 0）给自环边
            num_nodes = x.size(0)
            loop_attr = torch.zeros((num_nodes, edge_attr_no_loop.size(1)), device=x.device)
            # 将真实化学键的属性和自环的属性拼接在一起
            final_edge_attr = torch.cat([edge_attr_no_loop, loop_attr], dim=0)

        # 用最终对齐的 edge_index 和 edge_attr 进行消息传递
        # 即使 final_edge_attr 是 None，propagate 也能正常工作
        out = self.propagate(edge_index_with_loop, x=x, edge_attr=final_edge_attr)
        
        return self.nn(out)

    def message(self, x_j, edge_attr):
        # 增加一个防御性检查：如果边属性不存在，就只返回节点特征
        if edge_attr is None:
            return x_j
        return x_j + edge_attr


class MyGINLayer(nn.Module):
    """单层 GIN 的封装，包含边特征 Embedding 和 BatchNorm"""
    def __init__(self, num_edge_emb_list, emb_dim, batch_norm=True, activation=None):
        super(MyGINLayer, self).__init__()
        self.num_edge_emb_list = num_edge_emb_list
        # GIN 核心的“大脑” MLP (放大两倍再缩小)
        mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim)
        )
        
        # 实例化自定义的 GIN 卷积操作
        self.conv = MyGINEConv(mlp)
        
        # 边特征的 Embedding 层
        self.edge_embeddings = nn.ModuleList()
        for num_emb in num_edge_emb_list:
            emb_module = MLP(input_dim=num_emb, hidden_dim=emb_dim,
                             output_dim=emb_dim, num_layers=1)
            self.edge_embeddings.append(emb_module)

        self.bn = nn.BatchNorm1d(emb_dim) if batch_norm else None
        self.activation = activation

    def forward(self, x, edge_index, categorical_edge_feats):
        # 1. 翻译边特征 (如果有多个边特征类别，翻译后求和)
        edge_embeds = []
        # categorical_edge_feats 的形状通常是 [num_edges, num_edge_features]
        # 我们按照特征的列（特征种类）进行遍历
        for i in range(len(self.edge_embeddings)):
            # 提取第 i 种边特征，形状 [num_edges]
            feat_i = categorical_edge_feats[:, i]
            feat_i_one_hot = F.one_hot(feat_i, num_classes=self.num_edge_emb_list[i]).float()
            
            edge_embeds.append(self.edge_embeddings[i](feat_i_one_hot))
            # 翻译成 [num_edges, emb_dim] 的向量
           
        
        # 融合成一个综合的边特征表示
        edge_attr = torch.stack(edge_embeds, dim=0).sum(0)

        # 2. 执行核心的图卷积操作 (消息传递 + MLP 消化)
        out = self.conv(x, edge_index, edge_attr)

        # 3. 归一化与激活
        if self.bn is not None:
            out = self.bn(out)
        if self.activation is not None:
            out = self.activation(out)

        return out


# =====================================================================
# 3. 主干网络：MyGIN
# =====================================================================
class MyGIN(nn.Module):
    """
    完整的 GIN 网络。
    接收 PyG 的 Data/Batch 对象，输出图级别的特征表示。
    """
    def __init__(self, num_node_emb_list, num_edge_emb_list,
                 num_layers=5, emb_dim=300, JK='last', dropout=0., readout="mean"):
        super(MyGIN, self).__init__()
        self.num_node_emb_list = num_node_emb_list
        self.num_edge_emb_list = num_edge_emb_list
        self.num_layers = num_layers
        self.JK = JK
        self.dropout = nn.Dropout(dropout)
        
        # 检查参数
        if num_layers < 2:
            raise ValueError(f'Number of GNN layers must be > 1, got {num_layers}')
        if JK not in ['concat', 'last', 'max', 'sum']:
            raise ValueError(f"Expect JK to be 'concat', 'last', 'max' or 'sum', got {JK}")

        # 1. 节点特征的 Embedding 层
        self.node_embeddings = nn.ModuleList()
        for num_emb in num_node_emb_list:
            emb_module = MLP(input_dim=num_emb, hidden_dim=emb_dim,
                             output_dim=emb_dim, num_layers=2)
            self.node_embeddings.append(emb_module)

        # 2. 堆叠 GIN 卷积层
        self.gnn_layers = nn.ModuleList()
        for layer in range(num_layers):
            if layer == num_layers - 1:
                # 最后一层不加激活函数
                self.gnn_layers.append(MyGINLayer(num_edge_emb_list, emb_dim))
            else:
                self.gnn_layers.append(MyGINLayer(num_edge_emb_list, emb_dim, activation=F.relu))

        # 3. 图池化层 (Readout) 设置
        self.readout_type = readout

    def forward(self, x, edge_index, edge_attr, batch, graph_prompt=None):
        """
        参数:
        - x: 节点特征，形状 [num_nodes, num_node_features]
        - edge_index: 边索引，形状 [2, num_edges]
        - edge_attr: 边特征，形状 [num_edges, num_edge_features]
        - batch: 节点归属索引，形状 [num_nodes] (用于图池化)
        """
        # 1. 翻译节点特征 (如果有多个节点特征类别，翻译后求和)
        node_embeds = []
        for i in range(len(self.node_embeddings)):
            # 提取第 i 种节点特征
            feat_i = x[:, i]
            feat_i_one_hot = F.one_hot(feat_i, num_classes=self.num_node_emb_list[i]).float()
            node_embeds.append(self.node_embeddings[i](feat_i_one_hot))
        h = torch.stack(node_embeds, dim=0).sum(0)
        if graph_prompt is not None:
            h = h + graph_prompt[batch]

        # 2. 穿过每一层 GIN，并保存中间结果用于 JK
        all_layer_node_feats = [h]
        for layer in range(self.num_layers):
            h = self.gnn_layers[layer](h, edge_index, edge_attr)
            h = self.dropout(h)
            all_layer_node_feats.append(h)

        # 3. Jumping Knowledge (JK) 融合机制
        if self.JK == 'concat':
            final_node_feats = torch.cat(all_layer_node_feats, dim=1)
        elif self.JK == 'last':
            final_node_feats = all_layer_node_feats[-1]
        elif self.JK == 'max':
            stacked_feats = torch.stack(all_layer_node_feats, dim=0)
            final_node_feats = torch.max(stacked_feats, dim=0)[0]
        elif self.JK == 'sum':
            stacked_feats = torch.stack(all_layer_node_feats, dim=0)
            final_node_feats = torch.sum(stacked_feats, dim=0)

        # 4. 图池化 (Readout): 从节点特征得到图特征
        if self.readout_type == 'sum':
            graph_feats = global_add_pool(final_node_feats, batch)
        elif self.readout_type == 'mean':
            graph_feats = global_mean_pool(final_node_feats, batch)
        elif self.readout_type == 'max':
            graph_feats = global_max_pool(final_node_feats, batch)
        else:
            # 简化处理：对于复杂的 set2set 或 attention pooling，
            # 在目前的化学 OOD 任务中 mean pooling 通常已经足够好且最稳定。
            # 如果确有需要，可以引入 PyG 的 Set2Set 模块。
            raise ValueError(f"Readout '{self.readout_type}' is not fully supported in this simplified PyG version. Use 'mean' or 'sum'.")

        return graph_feats