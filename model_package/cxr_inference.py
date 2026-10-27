"""Self-contained inference for the chest X-ray image models"""
import io, json, sys
from pathlib import Path
import numpy as np, cv2, torch, torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms.v2.functional as TF
from PIL import Image

def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

class Net(nn.Module):
    def __init__(self, family, img_size=224):
        super().__init__(); self.family = family
        if family == "resnet50":
            self.backbone = tvm.resnet50(weights=None); D = self.backbone.fc.in_features; self.backbone.fc = nn.Identity()
        elif family == "densenet121":
            self.backbone = tvm.densenet121(weights=None); D = self.backbone.classifier.in_features; self.backbone.classifier = nn.Identity()
        elif family == "vit_small":
            import timm
            self.backbone = timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=False, num_classes=0, img_size=img_size); D = self.backbone.num_features
        else: raise ValueError(family)
        self.drop = nn.Dropout(0.0); self.fc = nn.Linear(D, 1)
    def forward(self, x): return self.fc(self.drop(self.backbone(x))).squeeze(-1)

class CXRPipeline:
    def __init__(self, folder, device=None):
        self.dir = Path(folder); self.cfg = json.loads((self.dir / "pipeline_config.json").read_text())
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        size = self.cfg["preprocessing"]["size"]; self.models = {}
        for f, m in self.cfg["models"].items():
            net = Net(f, size); net.load_state_dict(torch.load(self.dir / m["weights"], map_location="cpu")); self.models[f] = net.eval().to(self.device)
        self.mean = torch.tensor(self.cfg["normalisation"]["mean"]).view(1, 3, 1, 1); self.std = torch.tensor(self.cfg["normalisation"]["std"]).view(1, 3, 1, 1)

    @staticmethod
    def _open(src):
        if isinstance(src, Image.Image): return src
        if isinstance(src, (bytes, bytearray)): return Image.open(io.BytesIO(src))
        return Image.open(src)

    def preprocess(self, g):
        p = self.cfg["preprocessing"]
        if p["clahe"]: g = cv2.createCLAHE(clipLimit=p["clahe_clip"], tileGridSize=tuple(p["clahe_tile"])).apply(g)
        if p["resize"] != "squash": raise NotImplementedError("only the squash resize was used for training")
        return cv2.resize(g, (p["size"], p["size"]), interpolation=cv2.INTER_AREA)

    def _norm(self, x01): return (x01.repeat(1, 3, 1, 1) - self.mean) / self.std

    def qc_warnings(self, g, mode, size):
        w = []
        if mode not in ("L", "I;16"): w.append(f"file mode {mode} (training files were greyscale)")
        for k, v in dict(width=size[0], height=size[1], mean_int=float(g.mean()), std_int=float(g.std())).items():
            lo, hi = self.cfg["qc"][k]
            if not lo <= v <= hi: w.append(f"{k}={v:.0f} outside the training 1st-99th percentile ({lo:.0f}-{hi:.0f})")
        if g.std() < 10: w.append("almost blank image")
        return w

    def predict(self, src):
        im = self._open(src); g = np.asarray(im.convert("L")); warnings = self.qc_warnings(g, im.mode, im.size)
        a = self.preprocess(g); x01 = torch.from_numpy(a).float().div(255)[None, None]
        tta = [TF.affine(x01, angle=t["angle"], translate=[t["tx"], t["ty"]], scale=t["scale"], shear=[0., 0.]).mul(t["br"]).clamp(0, 1) for t in self.cfg["tta"]]
        xs = self._norm(torch.cat(tta)).to(self.device); x1 = self._norm(x01).to(self.device)
        Lk, L1, pf = [], [], {}
        with torch.no_grad():
            for f, net in self.models.items():
                T = self.cfg["models"][f]["temperature"]
                Lk.append(net(xs).float().cpu().numpy() / T)
                l1 = float(net(x1).float().cpu().numpy()[0]) / T; L1.append(l1); pf[f] = float(sigmoid(l1))
        Te = self.cfg["ensemble"]["temperature"]; thr = self.cfg["ensemble"]["threshold"]
        P = sigmoid(np.mean(Lk, 0) / Te); p, sd = float(P.mean()), float(P.std()); p_single = float(sigmoid(np.mean(L1) / Te))
        u = 1 - max(p, 1 - p) if self.cfg["uncertainty"]["score"] == "confidence" else sd
        return dict(p_pneumonia=p, predicted=self.cfg["labels"]["1" if p >= thr else "0"], threshold=thr, p_single_pass=p_single, per_model=pf,
                    tta_std=sd, uncertainty=float(u), low_confidence=bool(u > self.cfg["uncertainty"]["flag_threshold"]), warnings=warnings)

    def heatmap(self, src, pneumonia=None):
        from pytorch_grad_cam import GradCAMPlusPlus
        im = self._open(src); a = self.preprocess(np.asarray(im.convert("L")))
        x = self._norm(torch.from_numpy(a).float().div(255)[None, None]).to(self.device)
        if pneumonia is None: pneumonia = self.predict(src)["predicted"] == "PNEUMONIA"
        sign = 1.0 if pneumonia else -1.0; size = self.cfg["preprocessing"]["size"]; maps = []
        for f, net in self.models.items():
            layer = {"resnet50": lambda: [net.backbone.layer4[-1]], "densenet121": lambda: [net.backbone.features.norm5], "vit_small": lambda: [net.backbone.blocks[-1].norm1]}[f]()
            rt = (lambda t, h=size // 16: t[:, 1:, :].reshape(t.size(0), h, h, t.size(2)).permute(0, 3, 1, 2)) if f == "vit_small" else None
            with torch.enable_grad(), GradCAMPlusPlus(model=net, target_layers=layer, reshape_transform=rt) as cam:
                m = cam(input_tensor=x, targets=[lambda out, s=sign: s * out.reshape(-1)[0]])[0]
            maps.append((m - m.min()) / max(m.max() - m.min(), 1e-8))
        c = np.mean(maps, 0); return a, (c - c.min()) / max(c.max() - c.min(), 1e-8)

if __name__ == "__main__":
    print(json.dumps(CXRPipeline(Path(__file__).parent).predict(sys.argv[1]), indent=2))
