
import json
import os
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torchvision.models as tvm

from .preprocess import build_views


class Net(nn.Module):
    def __init__(self, family, img_size=224):
        super().__init__()

        if family == "resnet50":
            self.backbone = tvm.resnet50(weights=None)
            features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif family == "densenet121":
            self.backbone = tvm.densenet121(weights=None)
            features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()

        elif family == "vit_small":
            self.backbone = timm.create_model(
                "vit_small_patch16_224.augreg_in21k_ft_in1k",
                pretrained=False,
                num_classes=0,
                img_size=img_size,
            )
            features = self.backbone.num_features

        else:
            raise ValueError(f"Unsupported model: {family}")

        self.drop = nn.Dropout(0.0)
        self.fc = nn.Linear(features, 1)

    def forward(self, x):
        return self.fc(self.drop(self.backbone(x)))


class PneumoniaEnsemble:
    def __init__(self, package_dir=None, device=None):
        self.dir = Path(
            package_dir or os.getenv("MODEL_PACKAGE_DIR", "model_package")
        )

        self.cfg = json.loads(
            (self.dir / "pipeline_config.json").read_text()
        )

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.models = {
            name: self._load(name, config["weights"])
            for name, config in self.cfg["models"].items()
        }

        self.labels = {
            int(key): value
            for key, value in self.cfg["labels"].items()
        }

    def _load(self, name, relative_path):
        path = self.dir / relative_path
        model = Net(name, self.cfg["preprocessing"]["size"])

        state_dict = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        model.load_state_dict(state_dict, strict=True)

        return model.eval().to(self.device)

    @staticmethod
    def _two_logits(z):
        return torch.cat([torch.zeros_like(z), z], dim=1)

    @torch.inference_mode()
    def predict(self, raw: bytes, only=None):
        x = build_views(raw, self.cfg).to(self.device)
        names = only or list(self.models)

        per_model = {}
        calibrated_logits = []

        for name in names:
            temperature = self.cfg["models"][name]["temperature"]
            logits = self.models[name](x) / temperature

            two_class_logits = self._two_logits(logits)
            calibrated_logits.append(two_class_logits)

            probability = torch.softmax(two_class_logits, dim=1)[:, 1].mean()
            per_model[name] = float(probability)

        ensemble_temperature = self.cfg["ensemble"]["temperature"]

        mean_logits = torch.stack(calibrated_logits).mean(dim=0)
        probability = torch.softmax(
            mean_logits / ensemble_temperature, dim=1
        )[:, 1].mean().item()

        threshold = self.cfg["ensemble"]["threshold"]
        uncertain = (
            abs(probability - threshold)
            < self.cfg["uncertainty"]["flag_threshold"]
        )
        predicted_class = int(probability >= threshold)

        return {
            "prediction": self.labels[predicted_class],
            "probability_pneumonia": round(probability, 4),
            "threshold": threshold,
            "uncertain": uncertain,
            "per_model_probability": {
                key: round(value, 4)
                for key, value in per_model.items()
            },
            "tta_views": x.shape[0],
        }
