# Updated version of GridEvaluator class with fun-based flexible filename parsing
import os
import torch
import pandas as pd
from eval_module.grid_loss import GridLossEvaluator

def get_model_instance(pth_path, model_map: dict, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    filename = os.path.basename(pth_path)
    base_name = filename[:-4]

    # General fallback: everything before the last underscore
    parts = base_name.split("_")
    model_name = "_".join(parts[:-1])  # remove last token (usually loss/fp16 etc.)
    class_name = parts[0]  # remove last token (usually loss/fp16 etc.)
    if class_name not in model_map:
        raise ValueError(f"Unknown model name: {model_name} — 확인된 모델: {list(model_map.keys())}")

    model_class = model_map[class_name]
    model = model_class().to(device)
    state_dict = torch.load(pth_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

class GridEvaluator:
    def __init__(self, train_loader,val_loader, test_loader, checkpoint_dir,plot_dir,inference_dir,model_map, loss_fns=None, device=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.checkpoint_dir = checkpoint_dir
        self.plot_dir = plot_dir
        self.model_map = model_map
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fns = loss_fns
        self.results = []
        self.visualizer = None
        self.inference_dir = inference_dir

    def paths(self):
        path_list = os.listdir(self.checkpoint_dir)
        return [os.path.join(self.checkpoint_dir, path) for path in path_list if path.endswith('.pth')]

    def parse_name(self, filename):
        # Custom name parser with fallback for longer names
        parts = filename.replace(".pth", "").replace("fp16", "").replace("fp32", "").split("_")
        model_name = "_".join(parts[:-1])
        loss_name = parts[-1]
        return model_name, loss_name

    def run(self, percentile=0.90):
        for path in self.paths():
            model_name, loss_name = self.parse_name(os.path.basename(path))
            print(f"Evaluating model: {model_name} with loss: {loss_name}")
            model = get_model_instance(path, self.model_map, self.device)
            gridevaluator = GridLossEvaluator(
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                test_loader=self.test_loader,
                model=model,
                model_name=model_name,
                loss_fns=self.loss_fns,
                device=self.device,
                percentile=percentile,
                plot_dir=self.plot_dir,
                path=path,
                inference_dir=self.inference_dir,
            )
            df = gridevaluator.run()
            df['train_loss_name'] = loss_name
            self.results.append(df)
            print(f"Evaluation complete for model: {model_name} with loss: {loss_name}")

    def save_results(self, output_dir='./eval_results'):
        os.makedirs(output_dir, exist_ok=True)
        combined_df = pd.concat(self.results, ignore_index=True)
        combined_df.to_csv(os.path.join(output_dir, 'evaluation_results.csv'), index=False)
