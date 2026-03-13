import torch
import torch.nn as nn
import torch.nn.functional as F

class FusedMoE(nn.Module):
    def __init__(self, num_experts, hidden_size, intermediate_size, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.intermediate_size = intermediate_size

        # w13: [E, 2*I, H] -> 包含 Gate (w1) 和 Up (w3)
        self.w13 = nn.Parameter(torch.empty(num_experts, 2 * intermediate_size, hidden_size))
        # w2: [E, H, I] -> Down projection
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        
        nn.init.trunc_normal_(self.w13, std=0.02)
        nn.init.trunc_normal_(self.w2, std=0.02)

    def weight_loader(self, loaded_weight: torch.Tensor, 
                      expert_id: int, shard_id: str):
        """
        shard_id: 'w1', 'w2', or 'w3'
        """
        if expert_id >= self.num_experts:
            return

        with torch.no_grad():
            if shard_id == "w1":
                self.w13.data[expert_id, :self.intermediate_size, :].copy_(loaded_weight)
            elif shard_id == "w3":
                self.w13.data[expert_id, self.intermediate_size:, :].copy_(loaded_weight)
            elif shard_id == "w2":
                if self.w2.data[expert_id].shape != loaded_weight.shape:
                    loaded_weight = loaded_weight.view_as(self.w2.data[expert_id])
                self.w2.data[expert_id].copy_(loaded_weight)

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        # hidden_states: [N, H]
        # router_logits: [N, E]
        N, H = hidden_states.shape
        
        scores = F.softmax(router_logits, dim=-1)
        topk_weights, topk_ids = torch.topk(scores, self.top_k, dim=-1)
        if self.top_k > 1:
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(hidden_states)

        for e in range(self.num_experts):
            mask = (topk_ids == e)
            if not mask.any():
                continue

            token_indices, topk_indices = torch.where(mask)

            x_selected = hidden_states[token_indices] # [M, H]

            w13 = self.w13[e] # [2*I, H]
            gate_up = F.linear(x_selected, w13)
            gate, up = gate_up.chunk(2, dim=-1)
            h = F.silu(gate) * up
            
            y = F.linear(h, self.w2[e]) # [M, H]
            weight = topk_weights[token_indices, topk_indices].unsqueeze(-1)
            out.index_add_(0, token_indices, y * weight)

        return out