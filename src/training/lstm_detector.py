"""PyTorch LSTM-Autoencoder anomaly detector (model-swap alternative to IsolationForest, RQ4)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "n_handover", "n_unique_cells", "pingpong_count", "pingpong_rate",
    "mean_inter_ho_s", "std_inter_ho_s", "entropy_cell_seq",
]
SEQ_LEN = 1


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, encoding_dim: int = 4, seq_len: int = SEQ_LEN):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.LSTM(n_features, encoding_dim, batch_first=True)
        self.decoder = nn.LSTM(encoding_dim, encoding_dim, batch_first=True)
        self.head = nn.Linear(encoding_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, seq_len, n_features)
        _, (h, _) = self.encoder(x)
        z = h[-1].unsqueeze(1).repeat(1, self.seq_len, 1)  # (B, seq_len, enc)
        dec, _ = self.decoder(z)
        return self.head(dec)


def _to_seq(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x.reshape(x.shape[0], SEQ_LEN, x.shape[1]), dtype=torch.float32)


def train_lstm_ae(features_path: Path, out_model_dir: Path, *, random_state: int = 42,
                  epochs: int = 30, batch_size: int = 32, encoding_dim: int = 4,
                  threshold_percentile: float = 95.0, label_column: str = "label") -> dict[str, Any]:
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    out_model_dir = Path(out_model_dir)
    out_model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(features_path)
    x_raw = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    y = df[label_column].to_numpy() if label_column in df.columns else None
    x_tr, x_val, _, y_val = train_test_split(
        x_raw, y if y is not None else np.zeros(len(x_raw)),
        test_size=0.2, random_state=random_state,
        stratify=y if (y is not None and len(set(y.tolist())) > 1) else None,
    )
    scaler = StandardScaler().fit(x_tr)
    xt, xv = _to_seq(scaler.transform(x_tr)), _to_seq(scaler.transform(x_val))

    model = LSTMAutoencoder(len(FEATURE_COLS), encoding_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(xt.shape[0])
        for i in range(0, xt.shape[0], batch_size):
            b = xt[perm[i:i + batch_size]]
            opt.zero_grad()
            loss = loss_fn(model(b), b)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        tr_mse = ((model(xt) - xt) ** 2).mean(dim=(1, 2)).numpy()
        val_mse = ((model(xv) - xv) ** 2).mean(dim=(1, 2)).numpy()
    threshold = float(np.percentile(tr_mse, threshold_percentile))
    val_pred = (val_mse > threshold).astype(int)

    metrics: dict[str, float] = {"threshold": threshold, "mean_val_mse": float(val_mse.mean())}
    if y is not None and len(set(y_val.tolist())) > 1:
        metrics.update({
            "precision": float(precision_score(y_val, val_pred, zero_division=0)),
            "recall": float(recall_score(y_val, val_pred, zero_division=0)),
            "f1": float(f1_score(y_val, val_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_val, val_mse)),
            "pr_auc": float(average_precision_score(y_val, val_mse)),
        })

    model_path = out_model_dir / "model_lstm_ae.pt"
    torch.save(model.state_dict(), model_path)
    meta_path = out_model_dir / "lstm_ae_meta.joblib"
    joblib.dump({"feature_cols": FEATURE_COLS, "threshold": threshold,
                 "encoding_dim": encoding_dim, "scaler": scaler,
                 "state_dict_path": str(model_path)}, meta_path)
    # NOTE on the contract: "model_path" intentionally points to the joblib
    # meta-bundle (scaler + threshold + state_dict_path) -- that is the single
    # artifact to register/load and to pass to predict_lstm_ae(). The raw .pt
    # weights live under "state_dict_path".
    return {"model_path": str(meta_path), "state_dict_path": str(model_path),
            "metrics": metrics, "train_rows": int(len(x_tr)), "val_rows": int(len(x_val))}


def predict_lstm_ae(x: np.ndarray, model_meta_path: Path) -> tuple[np.ndarray, np.ndarray]:
    # meta is a trusted internal artifact produced by train_lstm_ae() in this same pipeline
    # (holds feature_cols, threshold, fitted StandardScaler) -> joblib.load is safe here.
    meta = joblib.load(model_meta_path)
    model = LSTMAutoencoder(len(meta["feature_cols"]), meta["encoding_dim"])
    # state_dict_path holds only tensors -> weights_only=True avoids arbitrary-code execution.
    model.load_state_dict(torch.load(meta["state_dict_path"], weights_only=True))
    model.eval()
    xs = _to_seq(meta["scaler"].transform(x))
    with torch.no_grad():
        mse = ((model(xs) - xs) ** 2).mean(dim=(1, 2)).numpy()
    return (mse > meta["threshold"]).astype(int), mse
