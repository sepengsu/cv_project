import torch
import torch.nn as nn
import torch.nn.functional as F

class KeyValueMemory(nn.Module):
    def __init__(self, mem_size=100, key_dim=128, value_dim=128, topk=3):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(mem_size, key_dim))
        self.values = nn.Parameter(torch.randn(mem_size, value_dim))
        self.topk = topk

    def forward(self, query):
        if query.dim() == 2:
            return self._match_vector(query)
        elif query.dim() == 4:
            return self._match_spatial(query)
        else:
            raise ValueError("Unsupported query shape")

    def _match_vector(self, query):
        q_norm = F.normalize(query, dim=1)
        k_norm = F.normalize(self.keys, dim=1)
        sim = torch.matmul(q_norm, k_norm.T)
        topk_sim, topk_idx = torch.topk(sim, self.topk, dim=1)
        weights = F.softmax(topk_sim, dim=1)
        selected_values = self.values[topk_idx]  # (B, topk, D)
        output = torch.sum(weights.unsqueeze(-1) * selected_values, dim=1)
        return output  # (B, D)

    def _match_spatial(self, query):
        B, C, H, W = query.shape
        q = query.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        q = F.normalize(q, dim=-1)
        k = F.normalize(self.keys, dim=1)
        sim = torch.matmul(q, k.T)  # (B, H*W, M)
        top_val, top_idx = torch.topk(sim, self.topk, dim=-1)  # (B, H*W, topk)
        weights = F.softmax(top_val, dim=-1)  # (B, H*W, topk)
        v_selected = self.values[top_idx]  # (B, H*W, topk, D)
        weighted = (v_selected * weights.unsqueeze(-1)).sum(dim=2)  # (B, H*W, D)
        return weighted.permute(0, 2, 1).view(B, C, H, W)


class HardMemory(nn.Module):
    def __init__(self, mem_size=100, mem_dim=128, threshold=0.8,topk=1):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(mem_size, mem_dim))
        self.threshold = threshold
        self.topk = topk

    def forward(self, x):
        if x.dim() == 2:
            x_norm = F.normalize(x, dim=1)
            mem_norm = F.normalize(self.memory, dim=1)
            sim = torch.matmul(x_norm, mem_norm.T)  # (B, M)
            max_val, max_idx = torch.max(sim, dim=1)  # (B,)

            matched = self.memory[max_idx]  # (B, D)

            # ⚠️ threshold filtering
            mask = (max_val > self.threshold).float().unsqueeze(-1)  # (B, 1)
            output = matched * mask  # similarity 낮으면 zero vector
            return output

        elif x.dim() == 4:
            B, C, H, W = x.shape
            x_flat = x.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
            x_norm = F.normalize(x_flat, dim=-1)
            mem_norm = F.normalize(self.memory, dim=1)

            sim = torch.matmul(x_norm, mem_norm.T)  # (B, H*W, M)
            max_val, max_idx = torch.max(sim, dim=-1)  # (B, H*W)

            mem_selected = self.memory[max_idx]  # (B, H*W, C)
            mask = (max_val > self.threshold).float().unsqueeze(-1)  # (B, H*W, 1)

            masked = mem_selected * mask  # 유사도 낮으면 0
            return masked.permute(0, 2, 1).view(B, C, H, W)

        else:
            raise ValueError("Unsupported input shape for HardMemory")
