from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SegmentAttentionBundle:
    state_dict: dict[str, Any]
    input_dimension: int
    hidden_dimension: int
    labels: list[str]
    training_config: dict[str, Any]


def fit_segment_attention_expert(
    train_sequence: np.ndarray,
    train_mask: np.ndarray,
    test_sequence: np.ndarray,
    test_mask: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    split_indices: list[tuple[np.ndarray, np.ndarray]],
    config: dict[str, Any],
) -> tuple[SegmentAttentionBundle, np.ndarray, np.ndarray, dict[str, Any]]:
    """Train a compact attention pooler on frozen mHuBERT windows."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    seed = int(config.get("seed", 20260821))
    hidden = int(config.get("hidden_dimension", 128))
    epochs = int(config.get("epochs", 30))
    batch_size = int(config.get("batch_size", 64))
    learning_rate = float(config.get("learning_rate", 0.001))
    weight_decay = float(config.get("weight_decay", 0.001))
    device_name = str(config.get("device", "cpu"))
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    label_index = {label: index for index, label in enumerate(labels)}
    targets = np.asarray([label_index[str(value)] for value in y], dtype=np.int64)
    input_dimension = int(train_sequence.shape[-1])

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(input_dimension, hidden)
            self.attention = nn.Linear(hidden, 1)
            self.classifier = nn.Linear(hidden, len(labels))

        def forward(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            valid = valid.clone()
            valid[~valid.any(dim=1), 0] = True
            projected = torch.tanh(self.projection(values))
            score = self.attention(projected).squeeze(-1)
            score = score.masked_fill(~valid, torch.finfo(score.dtype).min)
            weights = torch.softmax(score, dim=1)
            pooled = torch.sum(projected * weights.unsqueeze(-1), dim=1)
            return self.classifier(pooled)

    def train_model(indices: np.ndarray, fold_seed: int) -> Network:
        torch.manual_seed(fold_seed)
        np.random.seed(fold_seed)
        model = Network().to(device)
        counts = np.bincount(targets[indices], minlength=len(labels)).astype(float)
        class_weights = counts.sum() / np.maximum(counts * len(labels), 1.0)
        loss_function = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        dataset = TensorDataset(
            torch.from_numpy(train_sequence[indices]),
            torch.from_numpy(train_mask[indices]),
            torch.from_numpy(targets[indices]),
        )
        generator = torch.Generator().manual_seed(fold_seed)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        model.train()
        for _ in range(epochs):
            for values, valid, target in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(values.to(device), valid.to(device))
                loss = loss_function(logits, target.to(device))
                loss.backward()
                optimizer.step()
        return model.eval()

    def predict(model: Network, values: np.ndarray, valid: np.ndarray) -> np.ndarray:
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(values), batch_size):
                logits = model(
                    torch.from_numpy(values[start : start + batch_size]).to(device),
                    torch.from_numpy(valid[start : start + batch_size]).to(device),
                )
                outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    oof = np.zeros((len(train_sequence), len(labels)), dtype=float)
    fold_scores: list[dict[str, float]] = []
    for fold, (fit_index, validation_index) in enumerate(split_indices):
        model = train_model(fit_index, seed + fold)
        probability = predict(
            model, train_sequence[validation_index], train_mask[validation_index]
        )
        oof[validation_index] = probability
        predicted = probability.argmax(axis=1)
        fold_scores.append(
            {
                "fold": fold,
                "accuracy": float((predicted == targets[validation_index]).mean()),
            }
        )
    final_model = train_model(np.arange(len(train_sequence)), seed + 1000)
    test_probability = predict(final_model, test_sequence, test_mask)
    state_dict = {
        name: value.detach().cpu().numpy()
        for name, value in final_model.state_dict().items()
    }
    bundle = SegmentAttentionBundle(
        state_dict=state_dict,
        input_dimension=input_dimension,
        hidden_dimension=hidden,
        labels=labels,
        training_config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": seed,
            "device": device_name,
        },
    )
    metadata = {
        "fold_scores": fold_scores,
        "parameter_count": int(sum(parameter.numel() for parameter in final_model.parameters())),
        **bundle.training_config,
    }
    return bundle, oof, test_probability, metadata
