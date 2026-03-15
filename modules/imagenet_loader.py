import torch
import torch.nn as nn
import timm
from torchvision import models
import argparse
from collections import OrderedDict


class ImagenetLoader:
    REPLACEMENT_DROPOUT_RATE = 0.2
    REPLACEMENT_NUM_CLASSES = 61

    BACKBONE_OUT_FEATURES = {
        "resnet10t": 512,
        "resnet18": 512,
        "resnet50": 2048,
    }

    def __init__(self):
        self.classnum_to_label_map = None
        self.model = None

    def load_radimagenet_resnet50(self, weights_path=None, load_type="train"):
        self.model = models.resnet50(weights=None)
        self._setup_model(
            load_type, weights_path, self.BACKBONE_OUT_FEATURES["resnet50"]
        )
        return self.model

    def load_radimagenet_resnet18(self, weights_path=None, load_type="train"):
        self.model = models.resnet18(weights=None)
        self._setup_model(
            load_type, weights_path, self.BACKBONE_OUT_FEATURES["resnet18"]
        )
        return self.model

    def load_radimagenet_resnet10t(self, weights_path=None, load_type="train"):
        self.model = timm.create_model("resnet10t", pretrained=False)
        self._setup_model(
            load_type, weights_path, self.BACKBONE_OUT_FEATURES["resnet10t"]
        )
        return self.model

    def _setup_model(self, fc_type, weights_path, backbone_out_features):
        if fc_type == "train":
            self.model.fc = nn.Linear(backbone_out_features, 165)
            self._load_pretrained_weights(weights_path)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(p=self.REPLACEMENT_DROPOUT_RATE),
                nn.Linear(num_features, self.REPLACEMENT_NUM_CLASSES),
            )
        elif fc_type == "load":
            self.model.fc = nn.Sequential(
                nn.Dropout(p=self.REPLACEMENT_DROPOUT_RATE),
                nn.Linear(backbone_out_features, self.REPLACEMENT_NUM_CLASSES),
            )
            self._load_pretrained_weights(weights_path)

    def freeze_backbone(self, finetune_layer4=False):
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.fc.parameters():
            param.requires_grad = True

        if finetune_layer4:
            for param in self.model.layer4.parameters():
                param.requires_grad = True

    def get_trainable_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def _load_pretrained_weights(self, weights_path):
        torch.serialization.add_safe_globals([argparse.Namespace])
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

        state_dict = checkpoint["model"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("_orig_mod.", "")
            new_state_dict[name] = v

        self.model.load_state_dict(new_state_dict, strict=True)
