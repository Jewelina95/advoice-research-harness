from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


LABELS = ("HC", "MCI", "AD")


@dataclass
class ProtocolData:
    subject_ids: list[str]
    labels: np.ndarray
    partitions: np.ndarray
    texts: list[str]
    audio_sequence: np.ndarray
    audio_mask: np.ndarray
    states: np.ndarray
    reliability: np.ndarray
    task_index: np.ndarray
    language_index: np.ndarray
    age: np.ndarray
    task_categories: list[str]
    language_categories: list[str]
    state_features: list[str]


def _normalise_task(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def active_task_state_matrix(
    states: pd.DataFrame,
    task_by_subject: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Keep overall and observed-task states, not a sparse all-task matrix."""
    state_ids = sorted(
        column.removeprefix("state_")
        for column in states.columns
        if column.startswith("state_") and "__task_" not in column
    )
    values: list[list[float]] = []
    reliabilities: list[list[float]] = []
    for record in states.to_dict("records"):
        task = _normalise_task(task_by_subject.get(str(record["subject_id"]), "other"))
        state_row: list[float] = []
        reliability_row: list[float] = []
        for state_id in state_ids:
            overall = record.get(f"state_{state_id}", np.nan)
            overall_rel = record.get(f"rel_{state_id}", np.nan)
            task_value = record.get(f"state_{state_id}__task_{task}", np.nan)
            task_rel = record.get(f"rel_{state_id}__task_{task}", np.nan)
            state_row.extend(
                [overall, overall if pd.isna(task_value) else task_value]
            )
            reliability_row.extend(
                [overall_rel, overall_rel if pd.isna(task_rel) else task_rel]
            )
        values.append(state_row)
        reliabilities.append(reliability_row)
    names = [
        name
        for state_id in state_ids
        for name in (f"{state_id}_overall", f"{state_id}_active_task")
    ]
    value_matrix = np.nan_to_num(np.asarray(values, dtype=np.float32))
    reliability_matrix = np.clip(
        np.nan_to_num(np.asarray(reliabilities, dtype=np.float32)), 0.0, 1.0
    )
    keep: list[int] = []
    for overall_index in range(0, len(names), 2):
        active_index = overall_index + 1
        keep.append(overall_index)
        exact_duplicate = np.array_equal(
            value_matrix[:, overall_index], value_matrix[:, active_index]
        ) and np.array_equal(
            reliability_matrix[:, overall_index],
            reliability_matrix[:, active_index],
        )
        if not exact_duplicate:
            keep.append(active_index)
    return (
        value_matrix[:, keep],
        reliability_matrix[:, keep],
        [names[index] for index in keep],
    )


def protocol_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(LABELS)[probability.argmax(axis=1)]
    one_hot = np.column_stack([y_true == label for label in LABELS]).astype(int)
    numeric = np.asarray(
        [{label: index for index, label in enumerate(LABELS)}[str(value)] for value in y_true]
    )
    return {
        "accuracy": float(np.mean(predicted == y_true)),
        "micro_f1": float(f1_score(y_true, predicted, average="micro")),
        "macro_f1": float(f1_score(y_true, predicted, average="macro")),
        "micro_auroc_ovr": float(
            roc_auc_score(one_hot, probability, average="micro", multi_class="ovr")
        ),
        "macro_auroc_ovo": float(
            roc_auc_score(numeric, probability, average="macro", multi_class="ovo")
        ),
        "log_loss": float(log_loss(numeric, probability, labels=np.arange(len(LABELS)))),
    }


def apply_logit_offsets(probability: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    logits = np.log(np.clip(probability, 1e-8, 1.0)) + np.asarray(offsets, dtype=float)
    logits = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def fit_class_logit_offsets(
    y_true: np.ndarray,
    probability: np.ndarray,
    grid: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit class thresholds on development validation predictions only."""
    values = np.asarray(grid if grid is not None else np.arange(-1.0, 1.01, 0.1))
    best_offsets = np.zeros(len(LABELS), dtype=float)
    best_metrics = protocol_metrics(y_true, probability)
    best_key = (
        best_metrics["micro_f1"],
        best_metrics["macro_f1"],
        -best_metrics["log_loss"],
    )
    for mci_offset in values:
        for ad_offset in values:
            offsets = np.asarray([0.0, mci_offset, ad_offset])
            adjusted = apply_logit_offsets(probability, offsets)
            metrics = protocol_metrics(y_true, adjusted)
            key = (metrics["micro_f1"], metrics["macro_f1"], -metrics["log_loss"])
            if key > best_key:
                best_key = key
                best_offsets = offsets
                best_metrics = metrics
    return best_offsets, best_metrics


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _last_text_layers(model: Any, count: int) -> list[Any]:
    for layers in (
        getattr(getattr(model, "encoder", None), "layer", None),
        getattr(getattr(getattr(model, "encoder", None), "encoder", None), "layer", None),
    ):
        if layers is not None:
            return list(layers)[-count:]
    raise ValueError("Text encoder does not expose transformer encoder layers.")


def _network(
    text_model: Any,
    audio_dimension: int,
    state_dimension: int,
    task_count: int,
    language_count: int,
    config: dict[str, Any],
) -> Any:
    import torch
    from torch import nn

    hidden = int(config.get("hidden_dimension", 128))
    dropout = float(config.get("dropout", 0.15))
    maximum_windows = int(config["maximum_windows"])
    text_pooling = str(config.get("text_pooling", "mean"))
    explicit_position_ids = bool(config.get("explicit_position_ids", False))
    content_in_gate = bool(config.get("content_in_gate", True))
    gate_floor = float(config.get("gate_floor", 0.0))
    if gate_floor < 0.0 or gate_floor >= (1.0 / 3.0):
        raise ValueError("gate_floor must be in [0, 1/3).")

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.text_model = text_model
            self.text_head = nn.Sequential(
                nn.Linear(int(text_model.config.hidden_size), hidden),
                nn.Tanh(),
                nn.Dropout(dropout),
            )
            self.audio_projection = nn.Linear(audio_dimension, hidden)
            self.audio_cls = nn.Parameter(torch.zeros(1, 1, hidden))
            self.audio_position = nn.Parameter(
                torch.randn(1, maximum_windows + 1, hidden) * 0.02
            )
            layer = nn.TransformerEncoderLayer(
                hidden,
                4,
                hidden * 2,
                dropout,
                batch_first=True,
                norm_first=True,
            )
            self.audio_encoder = nn.TransformerEncoder(layer, 2)
            self.state_head = nn.Sequential(
                nn.Linear(state_dimension, hidden),
                nn.LayerNorm(hidden),
                nn.Tanh(),
                nn.Dropout(dropout),
            )
            context_dimension = max(16, hidden // 4)
            self.task_embedding = nn.Embedding(task_count, context_dimension)
            self.language_embedding = nn.Embedding(language_count, context_dimension)
            self.age_head = nn.Sequential(nn.Linear(2, context_dimension), nn.Tanh())
            self.context_conditioner = nn.Linear(
                context_dimension * 3, hidden * 6
            )
            self.classifiers = nn.ModuleList(
                [nn.Linear(hidden, len(LABELS)) for _ in range(3)]
            )
            self.gate = nn.Sequential(
                nn.Linear(
                    (hidden * 3 if content_in_gate else 0)
                    + context_dimension * 3
                    + 3,
                    hidden,
                ),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 3),
            )
            self.joint_classifier = nn.Sequential(
                nn.Linear(hidden * 6 + context_dimension * 3, hidden),
                nn.LayerNorm(hidden),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(hidden, len(LABELS)),
            )
            self.joint_gate = nn.Sequential(
                nn.Linear(context_dimension * 3 + 3, 1), nn.Sigmoid()
            )

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            audio: torch.Tensor,
            audio_mask: torch.Tensor,
            states: torch.Tensor,
            reliability: torch.Tensor,
            task: torch.Tensor,
            language: torch.Tensor,
            age: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
            text_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if explicit_position_ids:
                text_inputs["position_ids"] = torch.arange(
                    input_ids.shape[1], device=input_ids.device, dtype=torch.long
                ).unsqueeze(0).expand(input_ids.shape[0], -1)
            token_hidden = self.text_model(**text_inputs).last_hidden_state.float()
            if text_pooling == "cls":
                text_hidden = token_hidden[:, 0]
            elif text_pooling == "mean":
                token_mask = attention_mask.unsqueeze(-1).to(token_hidden.dtype)
                text_hidden = (token_hidden * token_mask).sum(dim=1) / token_mask.sum(
                    dim=1
                ).clamp_min(1.0)
            else:
                raise ValueError(f"Unsupported text_pooling={text_pooling!r}")
            text = self.text_head(text_hidden)
            audio_tokens = torch.tanh(self.audio_projection(audio))
            audio_tokens = torch.cat(
                [self.audio_cls.expand(len(audio), -1, -1), audio_tokens], dim=1
            )
            audio_tokens = audio_tokens + self.audio_position[:, : audio_tokens.shape[1]]
            valid = torch.cat(
                [
                    torch.ones((len(audio), 1), dtype=torch.bool, device=audio.device),
                    audio_mask,
                ],
                dim=1,
            )
            audio_representation = self.audio_encoder(
                audio_tokens, src_key_padding_mask=~valid
            )[:, 0]
            state = self.state_head(states)
            age_available = torch.isfinite(age).float().unsqueeze(1)
            age_value = torch.nan_to_num(age, nan=0.0).unsqueeze(1)
            context = torch.cat(
                [
                    self.task_embedding(task),
                    self.language_embedding(language),
                    self.age_head(torch.cat([age_value, age_available], dim=1)),
                ],
                dim=1,
            )
            raw_representations = torch.stack(
                [text, audio_representation, state], dim=1
            )
            condition = self.context_conditioner(context).view(
                len(text), 6, hidden
            )
            scale, shift = condition[:, :3], condition[:, 3:]
            representation_stack = raw_representations * (
                1.0 + 0.20 * torch.tanh(scale)
            ) + 0.20 * torch.tanh(shift)
            representations = list(representation_stack.unbind(dim=1))
            branch_logits = [
                classifier(representation)
                for classifier, representation in zip(
                    self.classifiers, representations, strict=True
                )
            ]
            gate_parts = (
                representations + [context, reliability]
                if content_in_gate
                else [context, reliability]
            )
            gate_input = torch.cat(gate_parts, dim=1)
            gate_logits = self.gate(gate_input) + torch.log(reliability.clamp_min(0.05))
            weights = torch.softmax(gate_logits, dim=1)
            if gate_floor:
                weights = gate_floor + (1.0 - 3.0 * gate_floor) * weights
            gated_logits = torch.sum(
                torch.stack(branch_logits, dim=1) * weights.unsqueeze(-1), dim=1
            )
            pairwise = [
                representations[0] * representations[1],
                representations[0] * representations[2],
                representations[1] * representations[2],
            ]
            joint_logits = self.joint_classifier(
                torch.cat(representations + pairwise + [context], dim=1)
            )
            # Keep the auditable branch mixture dominant while allowing the
            # joint head to model cross-branch interactions.
            joint_weight = 0.50 * self.joint_gate(
                torch.cat([context, reliability], dim=1)
            )
            logits = (1.0 - joint_weight) * gated_logits + joint_weight * joint_logits
            return logits, weights, branch_logits

    return Network()


def _predict(model: Any, loader: Any, device: Any) -> tuple[np.ndarray, np.ndarray]:
    import torch

    probabilities: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            logits, gate, _ = model(*[value.to(device) for value in batch[:-1]])
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            weights.append(gate.cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(weights)


def hierarchical_auxiliary_loss(logits: Any, target: Any, loss_function: Any) -> Any:
    """Supervise the HC/impaired and, conditionally, MCI/AD boundaries."""

    import torch
    from torch.nn import functional as functional

    screening_logits = torch.stack(
        [logits[:, 0], torch.logsumexp(logits[:, 1:], dim=1)], dim=1
    )
    screening_target = (target != 0).long()
    screening_loss = functional.cross_entropy(screening_logits, screening_target)
    impaired = target != 0
    if bool(impaired.any()):
        staging_loss = functional.cross_entropy(
            logits[impaired, 1:], target[impaired] - 1
        )
    else:
        staging_loss = logits.sum() * 0.0
    return 0.5 * (screening_loss + staging_loss)


def fit_protocol_model(
    data: ProtocolData,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    config: dict[str, Any],
    seed: int,
    fixed_epochs: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Fit a model without exposing any held-out test outcome to optimisation."""
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Subset, TensorDataset
    from transformers import AutoModel, AutoTokenizer

    _set_seed(seed)
    requested = str(config.get("device", "auto"))
    device_name = "mps" if requested == "auto" and torch.backends.mps.is_available() else requested
    device = torch.device("cpu" if device_name == "auto" else device_name)
    tokenizer = AutoTokenizer.from_pretrained(
        config["text_model"],
        revision=config["text_revision"],
        trust_remote_code=True,
    )
    text_prefix = str(config.get("text_prefix", ""))
    tokenized = tokenizer(
        [text_prefix + text for text in data.texts],
        padding="max_length",
        truncation=True,
        max_length=int(config.get("max_length", 512)),
        return_tensors="pt",
    )
    model_kwargs: dict[str, Any] = {
        "revision": config["text_revision"],
        "trust_remote_code": True,
        "attn_implementation": "eager",
    }
    if config.get("text_code_revision"):
        model_kwargs["code_revision"] = config["text_code_revision"]
    text_model = AutoModel.from_pretrained(
        config["text_model"], **model_kwargs
    ).float()
    for parameter in text_model.parameters():
        parameter.requires_grad = False
    for layer in _last_text_layers(
        text_model, int(config.get("trainable_text_layers", 2))
    ):
        for parameter in layer.parameters():
            parameter.requires_grad = True

    scaler = StandardScaler().fit(data.states[fit_indices])
    states = scaler.transform(data.states).astype(np.float32)
    reliability = np.column_stack(
        [
            np.ones(len(data.subject_ids), dtype=np.float32),
            data.audio_mask.any(axis=1).astype(np.float32),
            data.reliability.mean(axis=1).astype(np.float32),
        ]
    )
    age_mean = float(np.nanmean(data.age[fit_indices]))
    age_std = float(np.nanstd(data.age[fit_indices])) or 1.0
    age = ((data.age - age_mean) / age_std).astype(np.float32)
    label_index = {label: index for index, label in enumerate(LABELS)}
    targets = np.asarray([label_index[str(value)] for value in data.labels], dtype=np.int64)
    tensors = TensorDataset(
        tokenized["input_ids"],
        tokenized["attention_mask"],
        torch.from_numpy(data.audio_sequence.astype(np.float32)),
        torch.from_numpy(data.audio_mask.astype(bool)),
        torch.from_numpy(states),
        torch.from_numpy(reliability),
        torch.from_numpy(data.task_index.copy()),
        torch.from_numpy(data.language_index.copy()),
        torch.from_numpy(age),
        torch.from_numpy(targets),
    )

    def loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        return DataLoader(
            Subset(tensors, indices.tolist()),
            batch_size=int(config.get("batch_size", 8)),
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(seed),
            num_workers=0,
        )

    fit_loader = loader(fit_indices, True)
    validation_loader = loader(validation_indices, False)
    model = _network(
        text_model,
        data.audio_sequence.shape[-1],
        states.shape[1],
        len(data.task_categories),
        len(data.language_categories),
        config,
    ).to(device)
    backbone = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("text_model") and parameter.requires_grad
    ]
    heads = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("text_model") and parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone,
                "lr": float(config.get("text_learning_rate", 1e-6)),
                "weight_decay": float(config.get("weight_decay", 0.005)),
            },
            {
                "params": heads,
                "lr": float(config.get("head_learning_rate", 2e-4)),
                "weight_decay": float(config.get("weight_decay", 0.005)),
            },
        ]
    )
    class_weighting = str(config.get("class_weighting", "none"))
    if class_weighting == "balanced":
        fit_targets = targets[fit_indices]
        counts = np.bincount(fit_targets, minlength=len(LABELS)).astype(np.float32)
        weights = len(fit_targets) / (len(LABELS) * np.maximum(counts, 1.0))
        loss_function = nn.CrossEntropyLoss(
            weight=torch.from_numpy(weights).to(device)
        )
    elif class_weighting == "none":
        loss_function = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported class_weighting={class_weighting!r}")
    maximum_epochs = int(fixed_epochs or config.get("epochs", 14))
    patience = int(config.get("patience", 3))
    auxiliary_weight = float(config.get("auxiliary_loss_weight", 0.10))
    hierarchical_weight = float(config.get("hierarchical_loss_weight", 0.0))
    selection_metric = str(config.get("selection_metric", "micro_f1"))
    selection_mode = "min" if selection_metric == "log_loss" else "max"
    best_score = float("inf") if selection_mode == "min" else -float("inf")
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(maximum_epochs):
        model.train()
        losses: list[float] = []
        for batch in fit_loader:
            inputs = [value.to(device) for value in batch[:-1]]
            target = batch[-1].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _, branch_logits = model(*inputs)
            loss = loss_function(logits, target) + auxiliary_weight * sum(
                loss_function(branch, target) for branch in branch_logits
            ) / len(branch_logits)
            if hierarchical_weight:
                loss = loss + hierarchical_weight * hierarchical_auxiliary_loss(
                    logits, target, loss_function
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        probability, weights = _predict(model, validation_loader, device)
        metrics = protocol_metrics(data.labels[validation_indices], probability)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "validation": metrics,
                "mean_gate_weights": weights.mean(axis=0).tolist(),
            }
        )
        print(
            f"epoch={epoch + 1} train_loss={np.mean(losses):.4f} "
            f"val_loss={metrics['log_loss']:.4f} "
            f"val_micro_auc={metrics['micro_auroc_ovr']:.4f} "
            f"val_micro_f1={metrics['micro_f1']:.4f} "
            f"gate={weights.mean(axis=0).round(3).tolist()}",
            flush=True,
        )
        if fixed_epochs is not None:
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        else:
            score = float(metrics[selection_metric])
            improved = (
                score < best_score - 1e-4
                if selection_mode == "min"
                else score > best_score + 1e-4
            )
            if improved:
                best_score = score
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    probability, weights = _predict(model, validation_loader, device)
    metadata = {
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "validation": protocol_metrics(data.labels[validation_indices], probability),
        "validation_mean_gate_weights": weights.mean(axis=0).tolist(),
        "history": history,
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }
    model._advoice_scaler = scaler
    model._advoice_age = (age_mean, age_std)
    model._advoice_tokenizer = tokenizer
    return model, metadata


def predict_protocol_model(
    model: Any,
    data: ProtocolData,
    indices: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    tokenized = model._advoice_tokenizer(
        [str(config.get("text_prefix", "")) + data.texts[index] for index in indices],
        padding="max_length",
        truncation=True,
        max_length=int(config.get("max_length", 512)),
        return_tensors="pt",
    )
    states = model._advoice_scaler.transform(data.states[indices]).astype(np.float32)
    reliability = np.column_stack(
        [
            np.ones(len(indices), dtype=np.float32),
            data.audio_mask[indices].any(axis=1).astype(np.float32),
            data.reliability[indices].mean(axis=1).astype(np.float32),
        ]
    )
    age_mean, age_std = model._advoice_age
    age = ((data.age[indices] - age_mean) / age_std).astype(np.float32)
    dataset = TensorDataset(
        tokenized["input_ids"],
        tokenized["attention_mask"],
        torch.from_numpy(data.audio_sequence[indices].astype(np.float32)),
        torch.from_numpy(data.audio_mask[indices].astype(bool)),
        torch.from_numpy(states),
        torch.from_numpy(reliability),
        torch.from_numpy(data.task_index[indices]),
        torch.from_numpy(data.language_index[indices]),
        torch.from_numpy(age),
        torch.zeros(len(indices), dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=int(config.get("batch_size", 8)))
    return _predict(model, loader, next(model.parameters()).device)
