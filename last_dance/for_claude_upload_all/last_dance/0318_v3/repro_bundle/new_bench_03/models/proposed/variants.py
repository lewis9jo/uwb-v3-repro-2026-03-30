"""Proposed model variants for ablation studies."""
from __future__ import annotations

import torch.nn as nn

from ..registry import register_model
from .multitask_attention import ProposedUWBModel


@register_model("Proposed_V1")
class ProposedV1Model(ProposedUWBModel):
    """Original hyper-parameter profile."""

    def __init__(self, **kwargs) -> None:
        defaults = {
            "batch_size": 128,
            "epochs": 150,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "cls_weight": 2.0,
            "conf_weight": 0.5,
            "pos_weight": 2.5,
            "nlos_boost": 0.0,
        }
        for key, value in defaults.items():
            kwargs.setdefault(key, value)
        super().__init__(**kwargs)


@register_model("Proposed_V2")
class ProposedV2Model(ProposedUWBModel):
    """Doc-inspired variant with boosted NLOS weighting and longer training."""

    def __init__(self, **kwargs) -> None:
        defaults = {
            "batch_size": 96,
            "epochs": 200,
            "lr": 8e-4,
            "weight_decay": 2e-4,
            "patience": 45,
            "min_epochs": 60,
            "cls_weight": 2.5,
            "conf_weight": 0.7,
            "pos_weight": 3.0,
            "nlos_boost": 0.5,
        }
        for key, value in defaults.items():
            kwargs.setdefault(key, value)
        super().__init__(**kwargs)


@register_model("Proposed_NoAttention")
class ProposedNoAttentionModel(ProposedUWBModel):
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("use_attention", False)
        super().__init__(**kwargs)


@register_model("Proposed_NoMultitask")
class ProposedNoMultitaskModel(ProposedUWBModel):
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("use_multitask", False)
        kwargs.setdefault("cls_weight", 0.0)
        super().__init__(**kwargs)


@register_model("Proposed_NoConfidence")
class ProposedNoConfidenceModel(ProposedUWBModel):
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("use_confidence", False)
        kwargs.setdefault("conf_weight", 0.0)
        super().__init__(**kwargs)


@register_model("Proposed_NoSNR")
class ProposedNoSNRModel(ProposedUWBModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


@register_model("Proposed_8D")
class ProposedEightDimModel(ProposedUWBModel):
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("input_dim", 20)
        super().__init__(**kwargs)


# --------------------------
# V2-based ablations
# --------------------------


@register_model("Proposed_V2_NoAttention")
class ProposedV2NoAttentionModel(ProposedV2Model):
    """V2 hyperparameters with attention disabled."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("use_attention", False)
        super().__init__(**kwargs)


@register_model("Proposed_V2_NoMultitask")
class ProposedV2NoMultitaskModel(ProposedV2Model):
    """V2 hyperparameters with multitask/classification disabled."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("use_attention", False)
        kwargs.setdefault("use_multitask", False)
        kwargs.setdefault("cls_weight", 0.0)
        super().__init__(**kwargs)


@register_model("Proposed_V2_NoSNR")
class ProposedV2NoSNRModel(ProposedV2Model):
    """V2 hyperparameters without SNR input profile (uses base_no_snr profile)."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("use_attention", False)
        super().__init__(**kwargs)


@register_model("Proposed_V2_8D")
class ProposedV2EightDimModel(ProposedV2Model):
    """V2 hyperparameters with 8D input profile (base_8d)."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("use_attention", False)
        kwargs.setdefault("input_dim", 20)
        super().__init__(**kwargs)


@register_model("Proposed_V2_NoConfidence")
class ProposedV2NoConfidenceModel(ProposedV2Model):
    """V2 hyperparameters without confidence head/loss."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("use_attention", False)
        kwargs.setdefault("use_confidence", False)
        kwargs.setdefault("conf_weight", 0.0)
        super().__init__(**kwargs)


@register_model("Proposed_V4_Gated")
class ProposedV4GatedModel(ProposedV2Model):
    """V4: Proposed_V2 with attention replaced by simple sigmoid gating per branch."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _init_state(self):
        artifacts = super()._init_state()
        # gating over concatenated main/other (dim=80), independent weights per branch
        artifacts.model.attention = nn.Sequential(
            nn.Linear(80, 2),
            nn.Sigmoid(),
        )
        return artifacts
