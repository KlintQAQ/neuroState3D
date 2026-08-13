from __future__ import annotations

from typing import Literal

import torch.nn as nn

from models.generators.common import PosteriorBackboneConfig
from models.generators.diffusion import ConditionalLatentDiffusion, DiffusionConfig
from models.generators.drifting import ConditionalLatentDrifting, DriftingConfig


PosteriorMethod = Literal["diffusion", "drifting"]


def build_posterior_generator(
    method: PosteriorMethod,
    backbone: PosteriorBackboneConfig,
    *,
    diffusion: DiffusionConfig | None = None,
    drifting: DriftingConfig | None = None,
) -> nn.Module:
    """Build one of two parallel posterior engines with an identical API."""

    normalized = method.lower()
    if normalized == "diffusion":
        return ConditionalLatentDiffusion(backbone, diffusion or DiffusionConfig())
    if normalized == "drifting":
        return ConditionalLatentDrifting(backbone, drifting or DriftingConfig())
    raise ValueError("method must be 'diffusion' or 'drifting'")
