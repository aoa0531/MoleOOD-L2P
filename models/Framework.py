import torch
import math
from torch_geometric.nn import global_mean_pool
from .utils import pyg_batch_from_subgraphs, get_subgraph_batch_index
from .mygin import MyGIN
from .prompt_pool import PromptPool, SharedPrompt
# 【关键】从现在开始，所有 GNN Backbone (比如 GIN) 都必须是 PyG 的版本
# 这是一个示例的 PyG GIN 实现
from torch_geometric.nn import GIN, MLP

# ==========================================================
# 辅助模块 (保持不变，但更清晰)
# ==========================================================
class AttentionAgger(torch.nn.Module):
    def __init__(self, Qdim, Kdim, Mdim):
        super(AttentionAgger, self).__init__()
        self.model_dim = Mdim
        self.WQ = torch.nn.Linear(Qdim, Mdim)
        self.WK = torch.nn.Linear(Kdim, Mdim)

    def forward(self, Q, K, V, mask=None):
        # Q: [batch_size, Qdim], K/V: [num_subgraphs, Kdim]
        Q, K = self.WQ(Q), self.WK(K) # Q: [B, M], K: [S, M]
        
        Attn = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.model_dim) # [B, S]
        
        if mask is not None:
            # mask 的形状是 [B, S], True 的地方代表需要被掩盖
            Attn = Attn.masked_fill(mask, -1e9) # 用一个很大的负数填充
            
        Attn = torch.softmax(Attn, dim=-1) # [B, S]
        return torch.matmul(Attn, V) # [B, S] * [S, Vdim] -> [B, Vdim]

# ==========================================================
# 主角：Framework (核心改造)
# ==========================================================
class Framework(torch.nn.Module):
    def __init__(self, base_dim, sub_dim, num_class, dropout=0.5,
                 prompt_pool_size=None, prompt_init_scale=0.02):
        super(Framework, self).__init__()
        self.base_model = MyGIN(
              num_node_emb_list=[39], 
              num_edge_emb_list=[10], 
              num_layers=4,           
              emb_dim=base_dim, 
              JK='last', 
              dropout=0.1,            
              readout='mean'
          )
          
        self.sub_model = MyGIN(
            num_node_emb_list=[39], 
            num_edge_emb_list=[10], 
            num_layers=4, 
            emb_dim=sub_dim, 
            JK='last', 
            dropout=0.1, 
            readout='mean'
        )
        
        self.aggr = AttentionAgger(base_dim, sub_dim, max(base_dim, sub_dim))

        self.predictor = MLP([sub_dim, sub_dim, num_class], dropout=float(dropout), norm=None)
        if base_dim != sub_dim:
            raise ValueError('Additive graph/substructure fusion requires equal dimensions')
        self.base_dim = base_dim
        self.encoders_frozen = False
        self.prompt_pool = None
        if prompt_pool_size is not None:
            self.enable_prompts(prompt_pool_size, prompt_init_scale)


    def enable_prompts(self, pool_size=10, init_scale=0.02):
        if self.prompt_pool is not None:
            raise RuntimeError('Prompt pool already exists; do not reset it between stages')
        self.prompt_pool = PromptPool(pool_size, self.base_dim, init_scale).to(
            next(self.parameters()).device)
        self.freeze_encoders()

    def freeze_encoders(self):
        self.encoders_frozen = True
        for encoder in (self.base_model, self.sub_model):
            encoder.requires_grad_(False)
            encoder.eval()
        # The historical attention module is not used by this implementation.
        self.aggr.requires_grad_(False)

    def enable_shared_prompt(self, init_scale=0.02):
        if self.prompt_pool is not None:
            raise RuntimeError('Prompt module already exists')
        self.prompt_pool = SharedPrompt(self.base_dim, init_scale).to(next(self.parameters()).device)
        self.freeze_encoders()

    def train(self, mode=True):
        super().train(mode)
        if self.encoders_frozen:
            self.base_model.eval()
            self.sub_model.eval()
        return self

    def _sub_features(self, data, num_graphs):
        sub_batch = pyg_batch_from_subgraphs(data.subs)
        if sub_batch is None:
            return torch.zeros(num_graphs, self.base_dim, device=data.x.device)
        sub_batch = sub_batch.to(data.x.device)
        features = self.sub_model(sub_batch.x, sub_batch.edge_index,
                                  sub_batch.edge_attr, sub_batch.batch)
        owners = get_subgraph_batch_index(data.subs).to(data.x.device)
        summed = features.new_zeros(num_graphs, features.shape[-1])
        summed.index_add_(0, owners, features)
        counts = torch.bincount(owners, minlength=num_graphs).clamp_min(1)
        return summed / counts[:, None]

    def forward(self, data, return_aux=False):
        args = (data.x, data.edge_index, data.edge_attr, data.batch)
        indices = None
        if self.prompt_pool is None:
            main_feat = self.base_model(*args)
            key_loss = main_feat.new_zeros(())
        elif isinstance(self.prompt_pool, SharedPrompt):
            selected, indices, key_loss = self.prompt_pool.for_batch(data.num_graphs)
            main_feat = self.base_model(*args, graph_prompt=selected)
        else:
            with torch.no_grad():
                query = self.base_model(*args)
            selected, indices, key_loss = self.prompt_pool(query)
            # Frozen weights, but gradients through this branch reach the prompt.
            main_feat = self.base_model(*args, graph_prompt=selected)
        if self.encoders_frozen:
            with torch.no_grad():
                sub_feat = self._sub_features(data, main_feat.shape[0])
        else:
            sub_feat = self._sub_features(data, main_feat.shape[0])
        logits = self.predictor(main_feat + sub_feat)
        if return_aux:
            return {'logits': logits, 'key_loss': key_loss,
                    'prompt_indices': indices}
        return logits


# ==========================================================
# 配角 1：ConditionalGnn (改造)
# ==========================================================
class ConditionalGnn(torch.nn.Module):
    def __init__(self, emb_dim, backend_dim, num_domain, num_class):
        super(ConditionalGnn, self).__init__()
        self.class_emb = torch.nn.Embedding(num_domain, emb_dim)
        self.backend = MyGIN(
            num_node_emb_list=[39], 
            num_edge_emb_list=[10], 
            num_layers=4, 
            emb_dim=backend_dim, 
            JK='last', 
            dropout=0.1, 
            readout='mean'
        )
        self.predictor = MLP([backend_dim + emb_dim, backend_dim, num_class], norm=None)

    def forward(self, data, domains):
        domain_feat = self.class_emb(domains)
        graph_feat = self.backend(data.x, data.edge_index, data.edge_attr, data.batch)
      
        
        combined_feat = torch.cat([graph_feat, domain_feat], dim=1)
        return self.predictor(combined_feat)


# ==========================================================
# 配角 2：DomainClassifier (改造)
# ==========================================================
class DomainClassifier(torch.nn.Module):
    def __init__(self, backend_dim, num_domain, num_task=1):
        super(DomainClassifier, self).__init__()
        self.num_task = num_task 
        self.backend = MyGIN(
            num_node_emb_list=[39], 
            num_edge_emb_list=[10], 
            num_layers=4, 
            emb_dim=backend_dim, 
            JK='last', 
            dropout=0.1, 
            readout='mean'
        )
        self.predictor = MLP([backend_dim + num_task, backend_dim, num_domain], norm=None)

    def forward(self, data):
        graph_feat = self.backend(data.x, data.edge_index, data.edge_attr, data.batch)
       
        
        y_part = data.y.view(-1, self.num_task).float()
        
        combined_feat = torch.cat([graph_feat, y_part], dim=-1)
        return self.predictor(combined_feat)
