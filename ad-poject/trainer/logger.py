import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision

class LoggerMixin:
    def __init__(self, log_dir="./runs"):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_losses(self, train_loss, val_loss, epoch):
        self.writer.add_scalar("Loss/Train", train_loss, epoch)
        self.writer.add_scalar("Loss/Validation", val_loss, epoch)

    def log_gpu_usage(self, epoch):
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        self.writer.add_scalar("GPU/Memory Allocated (MB)", allocated, epoch)
        self.writer.add_scalar("GPU/Memory Reserved (MB)", reserved, epoch)

    def log_images(self, inputs, outputs, epoch, tag="Recon"):
        # inputs, outputs: (B, C, H, W)
        grid_input = torchvision.utils.make_grid(inputs[:8].cpu(), normalize=True)
        grid_output = torchvision.utils.make_grid(outputs[:8].cpu(), normalize=True)
        self.writer.add_image(f"{tag}/Input", grid_input, epoch)
        self.writer.add_image(f"{tag}/Output", grid_output, epoch)

    def close_logger(self):
        self.writer.close()
