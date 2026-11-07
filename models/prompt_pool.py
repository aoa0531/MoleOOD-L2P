"""Length-one graph prompts with independent, learnable retrieval keys."""
import torch
from torch import nn
from torch.nn import functional as F


class SharedPrompt(nn.Module):
    """One length-one prompt shared by every molecule; no keys or retrieval."""
    def __init__(self, dim, init_scale=0.02):
        super().__init__()
        self.prompts = nn.Parameter(torch.empty(1, dim))
        nn.init.uniform_(self.prompts, -init_scale, init_scale)

    def for_batch(self, batch_size):
        indices = torch.zeros(batch_size, dtype=torch.long, device=self.prompts.device)
        return self.prompts.expand(batch_size, -1), indices, self.prompts.new_zeros(())


class PromptPool(nn.Module):
    def __init__(self, pool_size, dim, init_scale=0.02):
        super().__init__()
        if pool_size < 1 or init_scale <= 0:
            raise ValueError('pool_size and init_scale must be positive')
        self.prompts = nn.Parameter(torch.empty(pool_size, dim))
        self.keys = nn.Parameter(torch.empty(pool_size, dim))
        nn.init.uniform_(self.prompts, -init_scale, init_scale)
        nn.init.uniform_(self.keys, -init_scale, init_scale)

    def forward(self, query):
        query = F.normalize(query.detach(), dim=-1)
        keys = F.normalize(self.keys, dim=-1)
        similarity = query @ keys.T
        indices = similarity.argmax(dim=-1)
        key_loss = (1 - (query * keys[indices]).sum(-1)).mean()
        return self.prompts[indices], indices, key_loss


def step_selected_rows(optimizer, pool, indices):
    """Protect inactive rows, including their Adam moments, during one step.

    Adam's scalar step counter is global; row-valued state is preserved exactly.
    The optimizer may also contain the predictor, which updates normally.
    """
    inactive = torch.ones(len(pool.prompts), dtype=torch.bool,
                          device=pool.prompts.device)
    inactive[indices.unique()] = False
    saved = []
    for parameter in (pool.prompts, pool.keys):
        state = optimizer.state.get(parameter, {})
        moments = {name: value[inactive].clone()
                   for name, value in state.items()
                   if torch.is_tensor(value) and value.shape == parameter.shape}
        saved.append((parameter, parameter.detach()[inactive].clone(), moments))
    optimizer.step()
    with torch.no_grad():
        for parameter, values, moments in saved:
            parameter[inactive] = values
            for name, value in optimizer.state.get(parameter, {}).items():
                if torch.is_tensor(value) and value.shape == parameter.shape:
                    value[inactive] = moments[name] if name in moments else 0
