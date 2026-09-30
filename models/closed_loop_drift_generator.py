from __future__ import annotations

import torch


def closed_loop_trainable_prefixes() -> tuple[str, ...]:
    return (
        "feedback_controller.",
        "token_drift_adapter.",
    )


def closed_loop_enabled(model: torch.nn.Module) -> bool:
    config = getattr(model, "config", None)
    return bool(getattr(config, "closed_loop_token_drift", False)) and all(
        hasattr(model, name)
        for name in ("feedback_controller", "token_drift_adapter")
    )


def closed_loop_parameter_report(model: torch.nn.Module) -> dict[str, int | bool]:
    prefixes = closed_loop_trainable_prefixes()
    closed_loop_parameters = 0
    trainable_closed_loop_parameters = 0
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            closed_loop_parameters += parameter.numel()
            if parameter.requires_grad:
                trainable_closed_loop_parameters += parameter.numel()
    return {
        "closed_loop_enabled": closed_loop_enabled(model),
        "closed_loop_parameters": int(closed_loop_parameters),
        "trainable_closed_loop_parameters": int(trainable_closed_loop_parameters),
    }
