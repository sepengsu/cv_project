import torch, os
from torch.utils.data import DataLoader
from tqdm import tqdm
from trainer.logger import LoggerMixin, GPUUsageLoggerMixin

def train(
    model,
    train_loader,
    val_loader=None,  # 🔥 default=None
    loss_fn=None,
    optimizer=None,
    scheduler=None,
    device='cuda',
    num_epochs=100,
    early_stop_patience=10,
    early_stopping_start_epoch=50,  # 🔥 추가: early stopping 시작 epoch
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
    torch.autograd.set_detect_anomaly(True)  # Gradient check

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)

        for x, _ in train_loop:
            x = x.to(device)
            output = model(x)
            loss = loss_fn(output, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        # 🔥 Validation Phase (val_loader 있을 때만)
        if val_loader is not None:
            model.eval()
            total_val_loss = 0.0
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
            with torch.no_grad():
                for x_val, _ in val_loop:
                    x_val = x_val.to(device)
                    output = model(x_val)
                    loss = loss_fn(output, x_val)
                    total_val_loss += loss.item()
                    val_loop.set_postfix(val_loss=loss.item())

            avg_val_loss = total_val_loss / len(val_loader)
            trainer.log_losses(avg_train_loss, avg_val_loss, epoch)
        else:
            avg_val_loss = None
            trainer.log_losses(avg_train_loss, None, epoch)

        trainer.log_gpu_usage(epoch)

        # 🔥 이미지 로깅 (10 epoch마다)
        if show_dataset and epoch % 10 == 0:
            with torch.no_grad():
                sample_x, label = next(iter(DataLoader(show_dataset, batch_size=16)))
                sample_x = sample_x.to(device)
                output = model(sample_x)
                if isinstance(output, tuple):
                    output = output[-1]
                trainer.log_images(sample_x, label, output, epoch)

        # ✅ Early Stopping (val_loader가 있을 때만)
        if val_loader is not None:
            if epoch >= early_stopping_start_epoch:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= early_stop_patience:
                        print("🛑 Early stopping triggered!")
                        break
            else:
                # Warmup 기간: best model 갱신
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()

        if scheduler:
            scheduler.step()

    print("Training complete!")
    if val_loader is not None:
        print(f"Best validation loss: {best_val_loss:.6f}")
    else:
        print(f"No validation was used during training.")

    # Save best model
    save_path = os.path.join(log_dir, "weight", "best_model.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if val_loader is not None and best_model_state is not None:
        torch.save(best_model_state, save_path)
        print(f"Model weights (best on val) saved to {save_path}")
    else:
        torch.save(model.state_dict(), save_path)
        print(f"Model weights (last epoch) saved to {save_path}")

    trainer.stop_gpu_monitor()
    trainer.save_gpu_peak_to_log(log_dir)
    trainer.close_logger()

    return model
