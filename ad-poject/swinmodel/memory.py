import torch
import torch.nn as nn
import torch.nn.functional as F

class KeyValueMemory(nn.Module):
    def __init__(self, mem_size=100, key_dim=128, value_dim=128, topk=3):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(mem_size, key_dim), requires_grad=False)  # 🔵 고정
        self.values = nn.Parameter(torch.randn(mem_size, value_dim), requires_grad=True) # 🔵 학습 가능
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
        return output

    def _match_spatial(self, query):
        B, C, H, W = query.shape
        q = query.view(B, C, -1).permute(0, 2, 1)
        q = F.normalize(q, dim=-1)
        k = F.normalize(self.keys, dim=1)
        sim = torch.matmul(q, k.T)
        top_val, top_idx = torch.topk(sim, self.topk, dim=-1)
        weights = F.softmax(top_val, dim=-1)
        v_selected = self.values[top_idx]
        weighted = (v_selected * weights.unsqueeze(-1)).sum(dim=2)
        return weighted.permute(0, 2, 1).view(B, C, H, W)

    def init_memory(self, feats):
        """
        feats: [N, D] tensor
        초기화는 key만 수정하고, value는 그대로 둔다
        """
        with torch.no_grad():
            n = min(self.keys.shape[0], feats.shape[0])
            self.keys[:n].copy_(feats[:n])


class HardMemory(nn.Module):
    def __init__(self, mem_size=100, mem_dim=128, threshold=0.8, topk=1):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(mem_size, mem_dim))
        self.threshold = threshold
        self.topk = topk

    def forward(self, x):
        if x.dim() == 2:
            return self._match_vector(x)
        elif x.dim() == 4:
            return self._match_spatial(x)
        else:
            raise ValueError("Unsupported input shape for HardMemory")

    def _match_vector(self, x):
        x_norm = F.normalize(x, dim=1)
        mem_norm = F.normalize(self.memory, dim=1)
        sim = torch.matmul(x_norm, mem_norm.T)
        max_val, max_idx = torch.max(sim, dim=1)

        matched = self.memory[max_idx]
        mask = (max_val > self.threshold).float().unsqueeze(-1)
        output = matched * mask
        return output

    def _match_spatial(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)
        x_norm = F.normalize(x_flat, dim=-1)
        mem_norm = F.normalize(self.memory, dim=1)

        sim = torch.matmul(x_norm, mem_norm.T)
        max_val, max_idx = torch.max(sim, dim=-1)

        mem_selected = self.memory[max_idx]
        mask = (max_val > self.threshold).float().unsqueeze(-1)
        masked = mem_selected * mask
        return masked.permute(0, 2, 1).view(B, C, H, W)

    def init_memory(self, feats):
        """
        feats: [N, D] tensor
        feats를 이용해서 self.memory를 초기화
        """
        with torch.no_grad():
            n = min(self.memory.shape[0], feats.shape[0])
            self.memory[:n].copy_(feats[:n])
