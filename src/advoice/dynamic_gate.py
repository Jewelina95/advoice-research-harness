from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DynamicGateBundle:
    state_dict: dict[str, Any]
    expert_count: int
    class_count: int
    hidden_dimension: int
    labels: list[str]
    training_config: dict[str, Any]


def fit_dynamic_reliability_gate(
    train_probability: np.ndarray,
    train_reliability: np.ndarray,
    test_probability: np.ndarray,
    test_reliability: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    split_indices: list[tuple[np.ndarray, np.ndarray]],
    config: dict[str, Any],
) -> tuple[
    DynamicGateBundle,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Cross-fit a compact case-specific gate over already cross-fitted experts."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    seed = int(config.get("seed", 20260821))
    hidden = int(config.get("hidden_dimension", 48))
    epochs = int(config.get("epochs", 60))
    batch_size = int(config.get("batch_size", 64))
    learning_rate = float(config.get("learning_rate", 0.002))
    weight_decay = float(config.get("weight_decay", 0.002))
    reliability_floor = float(config.get("reliability_floor", 0.05))
    device_name = str(config.get("device", "cpu"))
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)

    train_probability = np.asarray(train_probability, dtype=np.float32)
    test_probability = np.asarray(test_probability, dtype=np.float32)
    train_reliability = np.asarray(train_reliability, dtype=np.float32)
    test_reliability = np.asarray(test_reliability, dtype=np.float32)
    expert_count = int(train_probability.shape[1])
    class_count = int(train_probability.shape[2])
    if train_reliability.shape != train_probability.shape[:2]:
        raise ValueError("Expert reliability must have shape [subjects, experts].")
    if test_reliability.shape != test_probability.shape[:2]:
        raise ValueError("Test reliability must have shape [subjects, experts].")

    label_index = {label: index for index, label in enumerate(labels)}
    targets = np.asarray([label_index[str(value)] for value in y], dtype=np.int64)

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            context_dimension = expert_count * (class_count + 2)
            self.context = nn.Linear(context_dimension, hidden)
            self.gate = nn.Linear(hidden, expert_count)
            self.class_adjustment = nn.Linear(class_count, class_count)
            with torch.no_grad():
                self.class_adjustment.weight.copy_(torch.eye(class_count))
                self.class_adjustment.bias.zero_()

        def forward(
            self, probability: torch.Tensor, reliability: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            probability = probability.clamp_min(1e-7)
            probability = probability / probability.sum(dim=2, keepdim=True)
            reliability = reliability.clamp(0.0, 1.0)
            entropy = -torch.sum(probability * torch.log(probability), dim=2)
            entropy = entropy / np.log(max(class_count, 2))
            confidence = probability.max(dim=2).values
            context = torch.cat(
                [
                    probability.reshape(len(probability), -1),
                    reliability,
                    entropy,
                ],
                dim=1,
            )
            learned_gate = self.gate(torch.tanh(self.context(context)))
            learned_gate = learned_gate + torch.log(
                reliability.clamp_min(reliability_floor)
            )
            learned_gate = learned_gate + 0.25 * confidence
            weights = torch.softmax(learned_gate, dim=1)
            mixture = torch.sum(probability * weights.unsqueeze(-1), dim=1)
            logits = self.class_adjustment(torch.log(mixture.clamp_min(1e-7)))
            return logits, weights

    def train_model(indices: np.ndarray, fold_seed: int) -> Network:
        torch.manual_seed(fold_seed)
        np.random.seed(fold_seed)
        model = Network().to(device)
        counts = np.bincount(targets[indices], minlength=class_count).astype(float)
        class_weights = counts.sum() / np.maximum(counts * class_count, 1.0)
        loss_function = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), learning_rate, weight_decay=weight_decay
        )
        dataset = TensorDataset(
            torch.from_numpy(train_probability[indices]),
            torch.from_numpy(train_reliability[indices]),
            torch.from_numpy(targets[indices]),
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(fold_seed),
        )
        model.train()
        for _ in range(epochs):
            for probability, reliability, target in loader:
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(probability.to(device), reliability.to(device))
                loss = loss_function(logits, target.to(device))
                loss.backward()
                optimizer.step()
        return model.eval()

    def predict(
        model: Network, probability: np.ndarray, reliability: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        probabilities: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(probability), batch_size):
                logits, batch_weights = model(
                    torch.from_numpy(probability[start : start + batch_size]).to(device),
                    torch.from_numpy(reliability[start : start + batch_size]).to(device),
                )
                probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
                weights.append(batch_weights.cpu().numpy())
        return np.concatenate(probabilities), np.concatenate(weights)

    oof = np.zeros((len(train_probability), class_count), dtype=float)
    oof_weights = np.zeros((len(train_probability), expert_count), dtype=float)
    fold_scores: list[dict[str, float]] = []
    for fold, (fit_index, validation_index) in enumerate(split_indices):
        model = train_model(fit_index, seed + fold)
        probability, weights = predict(
            model,
            train_probability[validation_index],
            train_reliability[validation_index],
        )
        oof[validation_index] = probability
        oof_weights[validation_index] = weights
        fold_scores.append(
            {
                "fold": fold,
                "accuracy": float(
                    (probability.argmax(axis=1) == targets[validation_index]).mean()
                ),
            }
        )
    final_model = train_model(np.arange(len(train_probability)), seed + 1000)
    test_output, test_weights = predict(
        final_model, test_probability, test_reliability
    )
    state_dict = {
        name: value.detach().cpu().numpy()
        for name, value in final_model.state_dict().items()
    }
    training_config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "reliability_floor": reliability_floor,
        "seed": seed,
        "device": device_name,
    }
    bundle = DynamicGateBundle(
        state_dict=state_dict,
        expert_count=expert_count,
        class_count=class_count,
        hidden_dimension=hidden,
        labels=labels,
        training_config=training_config,
    )
    metadata = {
        "fold_scores": fold_scores,
        "parameter_count": int(
            sum(parameter.numel() for parameter in final_model.parameters())
        ),
        "mean_oof_weights": oof_weights.mean(axis=0).tolist(),
        **training_config,
    }
    return bundle, oof, test_output, oof_weights, test_weights, metadata
