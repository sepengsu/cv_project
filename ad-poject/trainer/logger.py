import torch, os, time, threading
from torch.utils.tensorboard import SummaryWriter
import torchvision

class LoggerMixin:
    def __init__(self, log_dir="./runs"):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_losses(self, train_loss, val_loss, epoch):
        self.writer.add_scalar("Loss/Train", train_loss, epoch)
        if val_loss is not None:
            self.writer.add_scalar("Loss/Validation", val_loss, epoch)


    def log_gpu_usage(self, epoch):
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        self.writer.add_scalar("GPU/Memory Allocated (MB)", allocated, epoch)
        self.writer.add_scalar("GPU/Memory Reserved (MB)", reserved, epoch)

    def log_images(self, inputs, labels, outputs, epoch, tag="Images"):
        """
        inputs, outputs: [B, C, H, W]
        labels: [B]
        """
        colored_inputs = []
        colored_outputs = []

        for x, y, label in zip(inputs, outputs, labels):
            colored_inputs.append(self.tint_image(x.cpu(), label.item()))
            colored_outputs.append(self.tint_image(y.cpu(), label.item()))

        grid_input = torchvision.utils.make_grid(colored_inputs, nrow=4, normalize=False)
        grid_output = torchvision.utils.make_grid(colored_outputs, nrow=4, normalize=False)

        self.writer.add_image(f"{tag}/Input", grid_input, epoch)
        self.writer.add_image(f"{tag}/Output", grid_output, epoch)


    def close_logger(self):
        self.writer.close()

    @staticmethod
    def tint_image(img_tensor, label):
        """
        img_tensor: [1, H, W] 흑백 이미지
        label: 0 or 1
        반환: [3, H, W] 컬러 이미지
        """
        img = img_tensor.expand(3, -1, -1).clone()  # [1, H, W] → [3, H, W]
        if label == 0:
            img[0] = 0  # R = 0 → 파란색
        else:
            img[1] = 0  # G = 0
            img[2] = 0  # B = 0 → 빨간색
        return img


class GPUUsageLoggerMixin:

    def __init__(self):
        self.gpu_peak_usage = 0
        self.gpu_monitor_start_time = None
        self.monitoring = False
        self.monitor_thread = None

    def _gpu_monitor_loop(self):
        """ 백그라운드에서 GPU peak usage 기록 """
        while self.monitoring:
            current = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            self.gpu_peak_usage = max(self.gpu_peak_usage, current)
            time.sleep(1)  # 1초 간격으로 체크

    def start_gpu_monitor(self):
        """ GPU 모니터링 시작 """
        self.gpu_peak_usage = 0
        self.gpu_monitor_start_time = time.time()
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._gpu_monitor_loop)
        self.monitor_thread.start()

    def stop_gpu_monitor(self):
        """ GPU 모니터링 종료 """
        self.monitoring = False
        self.monitor_thread.join()
        elapsed = time.time() - self.gpu_monitor_start_time
        print(f"[Summary] Max GPU Usage: {self.gpu_peak_usage:.2f} MB (Elapsed: {elapsed:.1f}s)")

    def save_gpu_peak_to_log(self, log_dir, filename="gpu_peak.log"):
        """ Peak GPU 사용량 저장 """
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, filename), "w") as f:
            f.write(f"Peak GPU Usage: {self.gpu_peak_usage:.2f} MB\n")
