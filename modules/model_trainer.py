import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from tqdm import tqdm
import os
import json
import time


class modelTrainer:
    def __init__(
        self,
        model,
        data_prep,
        device=None,
        learn_rate=0.001,
        num_epochs=10,
        model_name="model",
    ):
        self.model = model
        self.data_prep = data_prep
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.learn_rate = learn_rate
        self.num_epochs = num_epochs
        self.model_name = model_name

        self.optimizer = None
        self.loss_fn = None
        self.classnum_to_label_map = None

        self.history = {"train_loss": [], "train_f1": [], "val_loss": [], "val_f1": [], "epoch_time": []}

    def prepare_for_training(self, trainable_params=None):
        self.model = self.model.to(self.device)
        self.freeze_batch_norm()

        if trainable_params is None:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = optim.Adam(trainable_params, lr=self.learn_rate)

        class_weights = None
        if self.data_prep.class_weights is not None:
            class_weights = self.data_prep.class_weights.to(self.device)

        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        self.create_classnum_to_label_map(self.data_prep.class_names)

    def freeze_batch_norm(self):
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def train_epoch(self, epoch_num):
        self.model.train()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        prog_bar = tqdm(
            self.data_prep.train_loader,
            desc=f"Epoch {epoch_num+1}/{self.num_epochs}",
            unit="batch",
        )

        for images, labels in prog_bar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            prog_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.data_prep.train_loader)
        macro_f1 = f1_score(all_labels, all_preds, average="macro")

        return avg_loss, macro_f1

    def validate(self):
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        prog_bar = tqdm(
            self.data_prep.val_loader,
            desc="Validating",
            unit="batch",
        )

        with torch.no_grad():
            for images, labels in prog_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)

                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.data_prep.val_loader)
        macro_f1 = f1_score(all_labels, all_preds, average="macro")

        return avg_loss, macro_f1

    def save_model(self, current_epoch):
        model_dir = os.path.join("trained_models", self.model_name)
        os.makedirs(model_dir, exist_ok=True)

        checkpoint = {
            "model": self.model.state_dict(),
            "epoch": current_epoch,
            "loss_history": self.history.get("train_loss", []),
            "val_loss_history": self.history.get("val_loss", []),
            "train_f1_history": self.history.get("train_f1", []),
            "val_f1_history": self.history.get("val_f1", []),
            "epoch_time_history": self.history.get("epoch_time", []),
            "model_name": self.model_name,
        }

        torch.save(checkpoint, os.path.join(model_dir, f"{self.model_name}.pth"))

        if self.classnum_to_label_map:
            mapping_path = os.path.join(model_dir, "classnum_to_label_mapping.json")
            with open(mapping_path, "w") as f:
                json.dump(self.classnum_to_label_map, f)

    def load_model(self, path=None):
        load_path = path or os.path.join(
            "trained_models", self.model_name, f"{self.model_name}.pth"
        )
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))

    def train_all(self):
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            train_loss, train_f1 = self.train_epoch(epoch)
            val_loss, val_f1 = self.validate()
            epoch_time = time.time() - epoch_start_time

            self.history["train_loss"].append(train_loss)
            self.history["train_f1"].append(train_f1)
            self.history["val_loss"].append(val_loss)
            self.history["val_f1"].append(val_f1)
            self.history["epoch_time"].append(epoch_time)

            self.save_model(current_epoch=epoch + 1)

            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
            print(f"Epoch Time: {epoch_time:.2f}s\n")

    def create_classnum_to_label_map(self, class_names):
        self.classnum_to_label_map = {
            str(i): str(label) for i, label in enumerate(class_names)
        }
        return self.classnum_to_label_map
