import torch
import time
from sklearn.metrics import f1_score, recall_score
from tabulate import tabulate
import pandas as pd


class ModelEvaluator:
    def __init__(self, data_loader, class_names=None, device=None, silent=False):
        self.silent = silent
        self.data_loader = data_loader
        self.class_names = class_names
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}

    def evaluate_many(self, models_dict):
        for name, model in models_dict.items():
            print(f"\n[Evaluating] {name}...")
            self.evaluate_single(model, name)

        try:
            self.display_summary()
        except Exception as e:
            print(f"Summary failed due to: {e}")

    def evaluate_single(self, model, model_name="Model"):
        model.to(self.device)
        model.eval()

        all_preds = []
        all_labels = []

        # run 5 batches to "wake up" the GPU and allocate memory
        if not self.silent:
            print(f"\nWarming up {model_name}...")
        with torch.no_grad():
            for i, (images, _) in enumerate(self.data_loader):
                _ = model(images.to(self.device))
                if i >= 5:
                    break

        if not self.silent:
            print(f"Running inference...")
        total_time = 0.0
        num_samples = 0

        with torch.no_grad():
            for images, labels in self.data_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                if self.device == "cuda":
                    torch.cuda.synchronize()

                start_time = time.perf_counter()
                outputs = model(images)

                if self.device == "cuda":
                    torch.cuda.synchronize()
                end_time = time.perf_counter()

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                total_time += end_time - start_time
                num_samples += images.size(0)

        avg_latency_ms = (total_time / num_samples) * 1000

        metrics = {
            "f1_micro": f1_score(all_labels, all_preds, average="micro"),
            "f1_macro": f1_score(all_labels, all_preds, average="macro"),
            "sensitivity": recall_score(all_labels, all_preds, average="macro"),
            "avg_latency_ms": avg_latency_ms,
            "model_size_mb": self._get_model_size_mb(model),
            "device": self.device,
            "total_parameters": sum((p != 0).sum().item() for p in model.parameters())
        }

        self.results[model_name] = metrics
        return metrics

    def display_summary(self):
        summary_data = []
        for name, m in self.results.items():
            summary_data.append(
                [
                    name,
                    f"{m['f1_macro']:.2f}",
                    f"{m['avg_latency_ms']:.2f}",
                    f"{m['model_size_mb']:.2f}",
                ]
            )

        headers = [
            "Model",
            "F1 Macro",
            f"{self.device} Latency (ms)",
            "Model Size (mb)",
        ]
        print("\n" + "=" * 65)
        print("=" * 65)
        print(tabulate(summary_data, headers=headers, tablefmt="grid"))

    def _get_model_size_mb(self, model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
