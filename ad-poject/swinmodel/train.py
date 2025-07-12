import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from trainer.logger import LoggerMixin, GPUUsageLoggerMixin

def train(
    model,
    train_loader,
    val_loader,
    loss_fused,
    optimizer,
    scheduler=None,
    device='cuda',
    num_epochs=100,
    early_stop_patience=10,
    early_stopping_start_epoch=20,  # ✅ 추가된 부분: early stopping 시작 기준 epoch
    log_dir="./runs",
    show_dataset=None
):
    class Trainer(LoggerMixin, GPUUsageLoggerMixin):
        def __init__(self, log_dir):
            LoggerMixin.__init__(self, log_dir)
            GPUUsageLoggerMixin.__init__(self)

    trainer = Trainer(log_dir)
    trainer.start_gpu_monitor()

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        if hasattr(model, 'set_epoch'):
            model.set_epoch(epoch)

        total_train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{num_epochs}", leave=False)

        for x, _ in train_loop:
            x = x.to(device)
            output = model(x)
            if isinstance(output, tuple):
                output = output[-1]
            loss = loss_fused(output, x, epoch=epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_loop = tqdm(val_loader, desc=f"[Val] Epoch {epoch+1}/{num_epochs}", leave=False)
        with torch.no_grad():
            for x_val, _ in val_loop:
                x_val = x_val.to(device)
                output = model(x_val)
                if isinstance(output, tuple):
                    output = output[-1]
                val_loss = loss_fused(output, x_val, epoch=epoch)
                total_val_loss += val_loss.item()
                val_loop.set_postfix(val_loss=val_loss.item())

        avg_val_loss = total_val_loss / len(val_loader)
        trainer.log_losses(avg_train_loss, avg_val_loss, epoch)
        trainer.log_gpu_usage(epoch)

        # 🔥 이미지 로깅
        if show_dataset and epoch % 10 == 0:
            with torch.no_grad():
                sample_loader = DataLoader(show_dataset, batch_size=16, shuffle=True)
                sample_x, label = next(iter(sample_loader))
                sample_x = sample_x.to(device)
                output = model(sample_x)
                if isinstance(output, tuple):
                    output = output[-1]
                trainer.log_images(sample_x, label, output, epoch)

        # ✅ Early stopping with warmup
        if epoch >= early_stopping_start_epoch:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"🛑 Early stopping triggered at epoch {epoch+1}!")
                    break
        else:
            # Warmup 기간에는 무조건 best model 갱신만
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()

        if scheduler:
            scheduler.step()

    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Save best model
    weight_save_path = os.path.join(log_dir, "weight", "best_model.pth")
    os.makedirs(os.path.dirname(weight_save_path), exist_ok=True)
    torch.save(model.state_dict(), weight_save_path)
    print(f"Model weights saved to {weight_save_path}")

    trainer.stop_gpu_monitor()
    trainer.save_gpu_peak_to_log(log_dir)
    trainer.close_logger()

    return model
