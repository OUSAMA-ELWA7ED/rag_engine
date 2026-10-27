import cv2
import numpy as np
import torch


def decode_image(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image (expected PNG/JPEG).")
    return img


def base_preprocess(gray: np.ndarray, cfg: dict) -> np.ndarray:
    """grayscale -> CLAHE -> squash resize. Returns uint8 HxW."""
    pp = cfg["preprocessing"]
    if pp.get("clahe"):
        clahe = cv2.createCLAHE(clipLimit=pp["clahe_clip"], tileGridSize=tuple(pp["clahe_tile"]))
        gray = clahe.apply(gray)
    s = pp["size"]
    return cv2.resize(gray, (s, s), interpolation=cv2.INTER_AREA)  # "squash" = ignore aspect ratio


def apply_tta(img: np.ndarray, t: dict) -> np.ndarray:
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), t["angle"], t["scale"])
    M[:, 2] += (t["tx"], t["ty"])
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return np.clip(out.astype(np.float32) * t["br"], 0, 255).astype(np.uint8)


def to_tensor(img: np.ndarray, cfg: dict) -> torch.Tensor:
    x = img.astype(np.float32) / 255.0
    x = np.stack([x, x, x], 0)  # 3xHxW
    mean = np.array(cfg["normalisation"]["mean"], np.float32)[:, None, None]
    std = np.array(cfg["normalisation"]["std"], np.float32)[:, None, None]
    return torch.from_numpy((x - mean) / std)


def build_views(raw: bytes, cfg: dict) -> torch.Tensor:
    """Original + all TTA views -> (N,3,H,W)."""
    base = base_preprocess(decode_image(raw), cfg)
    views = [base] + [apply_tta(base, t) for t in cfg.get("tta", [])]
    return torch.stack([to_tensor(v, cfg) for v in views])
