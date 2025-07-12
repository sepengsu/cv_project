import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ Rearrange 함수 (einops 대체)
def rearrange(x, pattern, **axes):
    if pattern == 'b c h w -> b h w c':
        return x.permute(0, 2, 3, 1)
    elif pattern == 'b h w c -> b c h w':
        return x.permute(0, 3, 1, 2)
    elif pattern.startswith('b (nh ws1) (nw ws2) c -> (b nh nw) (ws1 ws2) c'):
        ws1 = axes['ws1']
        ws2 = axes['ws2']
        b, nh_ws1, nw_ws2, c = x.shape
        nh = nh_ws1 // ws1
        nw = nw_ws2 // ws2
        x = x.view(b, nh, ws1, nw, ws2, c)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(b * nh * nw, ws1 * ws2, c)
        return x
    elif pattern.startswith('(b nh nw) (ws1 ws2) c -> b (nh ws1) (nw ws2) c'):
        nh = axes['nh']
        nw = axes['nw']
        ws1 = axes['ws1']
        ws2 = axes['ws2']
        bnhnw, wsize, c = x.shape
        b = bnhnw // (nh * nw)
        x = x.view(b, nh, nw, ws1, ws2, c)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, nh * ws1, nw * ws2, c)
        return x
    else:
        raise NotImplementedError(f"Pattern '{pattern}' is not supported.")

# ✅ Padding & Cropping Functions
def pad_to_32(x):
    return F.pad(x, (2, 2, 2, 2))

def crop_to_28(x):
    return x[:, :, 2:-2, 2:-2]

# ✅ Patch Embedding
class PatchEmbed(nn.Module):
    def __init__(self, in_ch=1, embed_dim=64, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x)

# ✅ Window Attention
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=4, num_heads=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, H, W, C = x.shape
        assert H % self.window_size == 0 and W % self.window_size == 0

        x = rearrange(x, 'b (nh ws1) (nw ws2) c -> (b nh nw) (ws1 ws2) c',
                      ws1=self.window_size, ws2=self.window_size)

        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(t.shape[0], t.shape[1], self.num_heads, -1).transpose(1, 2), qkv)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / (q.shape[-1] ** 0.5))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        out = self.proj(out)

        out = rearrange(out, '(b nh nw) (ws1 ws2) c -> b (nh ws1) (nw ws2) c',
                        nh=H // self.window_size, nw=W // self.window_size, ws1=self.window_size, ws2=self.window_size)
        return out

# ✅ Swin Block
class SwinBlock(nn.Module):
    def __init__(self, dim, window_size=4, shift=False):
        super().__init__()
        self.shift = shift
        self.attn = WindowAttention(dim, window_size=window_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b h w c')

        if self.shift:
            shift = self.attn.window_size // 2
            x = torch.roll(x, shifts=(-shift, -shift), dims=(1, 2))

        x = self.attn(x)

        if self.shift:
            shift = self.attn.window_size // 2
            x = torch.roll(x, shifts=(shift, shift), dims=(1, 2))

        x = rearrange(x, 'b h w c -> b c h w')
        return x

# ✅ Swin Encoder
class SwinStructureEncoder(nn.Module):
    def __init__(self, in_ch=1, embed_dim=64, window_size=4):
        super().__init__()
        self.patch = PatchEmbed(in_ch, embed_dim, patch_size=window_size)
        self.block1 = SwinBlock(embed_dim, window_size=window_size, shift=False)
        self.block2 = SwinBlock(embed_dim, window_size=window_size, shift=True)

    def forward(self, x):
        x = self.patch(x)
        x = self.block1(x)
        x = self.block2(x)
        return x  # (B, C, H, W)

# ✅ Memory Module (Spatial)
class KeyValueMemory(nn.Module):
    def __init__(self, mem_size=400, dim=64, topk=5):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(mem_size, dim), requires_grad=False)
        self.values = nn.Parameter(torch.randn(mem_size, dim), requires_grad=False)
        self.topk = topk

    def forward(self, q):
        if q.dim() == 2:
            return self._match_vector(q)
        elif q.dim() == 4:
            return self._match_spatial(q)
        else:
            raise ValueError("Unsupported query shape")

    def _match_vector(self, q):
        q = F.normalize(q, dim=1)
        k = F.normalize(self.keys, dim=1)
        sim = torch.matmul(q, k.T)
        top_val, top_idx = torch.topk(sim, self.topk, dim=1)
        weights = F.softmax(top_val, dim=1)
        v_selected = self.values[top_idx]
        weighted = (v_selected * weights.unsqueeze(-1)).sum(dim=1)
        return weighted.unsqueeze(-1).unsqueeze(-1)

    def _match_spatial(self, q):
        B, C, H, W = q.shape
        q = q.view(B, C, -1).permute(0, 2, 1)
        q = F.normalize(q, dim=-1)
        k = F.normalize(self.keys, dim=1)
        sim = torch.matmul(q, k.T)
        top_val, top_idx = torch.topk(sim, self.topk, dim=-1)
        weights = F.softmax(top_val, dim=-1)
        v_selected = self.values[top_idx]
        weighted = (v_selected * weights.unsqueeze(-1)).sum(dim=2)
        return weighted.permute(0, 2, 1).view(B, C, H, W)

# ✅ Decoder (spatial match 기반, 입력 shape 유지)
class SimpleDecoder(nn.Module):
    def __init__(self, in_ch=64, out_ch=1):
        super().__init__()
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2), nn.ReLU(),  # 8 → 16
            nn.ConvTranspose2d(in_ch // 2, out_ch, kernel_size=2, stride=2)             # 16 → 32
        )

    def forward(self, x):
        return self.decode(x)  # (B, 1, 32, 32)



# ✅ Full Model
class SwinMemAE(nn.Module):
    def __init__(self, embed_dim=64, mem_size=400):
        super().__init__()
        self.encoder = SwinStructureEncoder(in_ch=1, embed_dim=embed_dim)
        self.memory = KeyValueMemory(mem_size, dim=embed_dim, topk=5)
        self.decoder = SimpleDecoder(in_ch=embed_dim, out_ch=1)

    def forward(self, x):
        x_32 = x.clone()
        x_32 = pad_to_32(x_32)
        feat = self.encoder(x_32)         # (B, C, H, W)
        mem_feat = self.memory(feat)      # spatial match → (B, C, H, W)
        x_hat = self.decoder(mem_feat)    # (B, 1, 32, 32)
        return crop_to_28(x_hat)          # (B, 1, 28, 28)

# ✅ Memory Initialization
@torch.no_grad()
def init_memory(model, dataloader, device='cuda', top_n=400):
    model.eval()
    memory_feats = []
    for x, _ in dataloader:
        x = pad_to_32(x.to(device))
        feat = model.encoder(x).detach().cpu()
        feat = feat.mean(dim=[2, 3])  # GAP for memory key
        memory_feats.append(feat)
    all_feats = torch.cat(memory_feats, dim=0)[:top_n]
    model.memory.keys.data.copy_(all_feats.clone().to(device))
    model.memory.values.data.copy_(all_feats.clone().to(device))
    model.memory.keys.requires_grad = True
    model.memory.values.requires_grad = True
    print(f"[✅] Initialized memory with {top_n} samples and enabled training.")

