import torch
import torch.nn.functional as F

# ⬇️ 중심 가중치 맵 생성 함수
def create_center_weight_map(size=28, sigma=5):
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    dist = torch.sqrt(grid_x**2 + grid_y**2)
    weight = torch.exp(- (dist ** 2) / (2 * (sigma / size) ** 2))
    return weight.unsqueeze(0).unsqueeze(0)  # shape: (1,1,H,W)

# ⬇️ 중심 crop 기반 loss
def center_crop_loss(x, y, size=16):
    _, _, H, W = x.shape
    assert size <= H and size <= W, "Crop size must be smaller than input size"
    start_h = (H - size) // 2
    start_w = (W - size) // 2
    x_c = x[:, :, start_h:start_h+size, start_w:start_w+size]
    y_c = y[:, :, start_h:start_h+size, start_w:start_w+size]
    raw = F.mse_loss(x_c, y_c, reduction='none')
    return raw

# ⬇️ 중심 가중치 기반 loss
def center_weighted_loss(x, y, weight_map):
    weight = weight_map.to(x.device)
    raw = ((x - y) ** 2) * weight
    return raw
