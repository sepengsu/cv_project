import torch 

# ✅ Memory 초기화 함수
# ✅ Memory 초기화 함수
@torch.no_grad()
def init_memory(model, dataloader, device='cuda', top_n=400):
    """
    SwinInspiredAEMemSkipV2 모델 전용 Memory 초기화 함수
    각 단계별 feature 평균을 모아 Memory에 저장함
    """
    model.eval()
    latent_feats, e3_feats, e2_feats, e1_feats = [], [], [], []

    for x, _ in dataloader:
        x = x.to(device)
        x = model.pad(x)
        e1 = model.enc1(x)
        e2 = model.enc2(e1)
        e3 = model.enc3(e2)
        e4 = model.enc4(e3)

        z_flat = model.flatten(e4)
        z_latent = model.latent_fc(z_flat)
        latent_feats.append(z_latent.cpu())

        e3_flat = e3.view(e3.size(0), e3.size(1), -1).mean(dim=-1)
        e3_feats.append(e3_flat.cpu())

        e2_flat = e2.view(e2.size(0), e2.size(1), -1).mean(dim=-1)
        e2_feats.append(e2_flat.cpu())

        e1_flat = e1.view(e1.size(0), e1.size(1), -1).mean(dim=-1)
        e1_feats.append(e1_flat.cpu())

        if len(latent_feats) * x.size(0) >= top_n:
            break

    latent_feats = torch.cat(latent_feats, dim=0)
    e3_feats = torch.cat(e3_feats, dim=0)
    e2_feats = torch.cat(e2_feats, dim=0)
    e1_feats = torch.cat(e1_feats, dim=0)

    with torch.no_grad():
        model.memory_latent.memory.copy_(latent_feats[:model.memory_latent.memory.size(0)])
        model.memory_e3.memory.copy_(e3_feats[:model.memory_e3.memory.size(0)])
        model.memory_e2.memory.copy_(e2_feats[:model.memory_e2.memory.size(0)])
        model.memory_e1.memory.copy_(e1_feats[:model.memory_e1.memory.size(0)])

    print(f"[✔] Memory Initialized: latent={latent_feats.size(0)}, e3={e3_feats.size(0)}, e2={e2_feats.size(0)}, e1={e1_feats.size(0)}")
