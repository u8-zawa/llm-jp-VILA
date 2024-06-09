# coding=utf-8

# Copyright 2024 LY Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from typing import Any, Literal, Optional

from transformers import PretrainedConfig


class CLYPConfig(PretrainedConfig):
    model_type = "clyp"

    def __init__(
        self,
        vision_encoder_config: Optional[dict] = None,
        text_encoder_config: Optional[dict] = None,
        itc_loss_config: Optional[dict] = None,
        learn_temperature: bool = True,
        temperature_init: float = 0.07,
        temperature_min: float = 0.01,
        temperature_max: float = 1000.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        vision_encoder_config = vision_encoder_config or {}
        text_encoder_config = text_encoder_config or {}
        self.vision_encoder_config = CLYPVisionEncoderConfig(**vision_encoder_config)
        self.text_encoder_config = CLYPTextEncoderConfig(**text_encoder_config)
        self.itc_loss_config = (
            CLYPLossConfig(**itc_loss_config) if itc_loss_config else None
        )
        self.learn_temperature = learn_temperature
        self.temperature_init = temperature_init
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max

    def to_diff_dict(self) -> dict[str, Any]:
        serializable_config_dict = super().to_diff_dict()
        sub_serializable_config_dict = {
            "vision_encoder_config": _to_diff_dict(self.vision_encoder_config),
            "text_encoder_config": _to_diff_dict(self.text_encoder_config),
        }
        self.dict_torch_dtype_to_str(sub_serializable_config_dict)
        serializable_config_dict.update(sub_serializable_config_dict)
        return serializable_config_dict


class CLYPVisionEncoderConfig(PretrainedConfig):
    def __init__(
        self,
        backbone_config: Optional[dict] = None,
        pooler_config: Optional[dict] = None,
        neck_config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        backbone_config = backbone_config or {}
        pooler_config = pooler_config or {"input_type": "timm"}
        neck_config = neck_config or {}
        self.backbone_config = CLYPVisionBackboneConfig(**backbone_config)
        self.pooler_config = CLYPPoolerConfig(**pooler_config)
        self.neck_config = CLYPNeckConfig(**neck_config)

    def to_diff_dict(self) -> dict[str, Any]:
        serializable_config_dict = {
            "backbone_config": _to_diff_dict(self.backbone_config),
            "pooler_config": _to_diff_dict(self.pooler_config),
            "neck_config": _to_diff_dict(self.neck_config),
        }
        self.dict_torch_dtype_to_str(serializable_config_dict)
        return serializable_config_dict


class CLYPTextEncoderConfig(PretrainedConfig):
    def __init__(
        self,
        backbone_config: Optional[dict] = None,
        pooler_config: Optional[dict] = None,
        neck_config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        backbone_config = backbone_config or {}
        pooler_config = pooler_config or {"input_type": "huggingface"}
        neck_config = neck_config or {}
        self.backbone_config = CLYPTextBackboneConfig(**backbone_config)
        self.pooler_config = CLYPPoolerConfig(**pooler_config)
        self.neck_config = CLYPNeckConfig(**neck_config)

    def to_diff_dict(self) -> dict[str, Any]:
        serializable_config_dict = {
            "backbone_config": _to_diff_dict(self.backbone_config),
            "pooler_config": _to_diff_dict(self.pooler_config),
            "neck_config": _to_diff_dict(self.neck_config),
        }
        self.dict_torch_dtype_to_str(serializable_config_dict)
        return serializable_config_dict


class CLYPVisionBackboneConfig(PretrainedConfig):
    def __init__(
        self,
        model_name: str = "eva02_base_patch16_clip_224.merged2b",
        pretrained: bool = True,
        extra_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name
        self.pretrained = pretrained
        self.extra_kwargs = extra_kwargs or {}


class CLYPTextBackboneConfig(PretrainedConfig):
    def __init__(
        self,
        model_name: str = "rinna/japanese-clip-vit-b-16",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name


class CLYPPoolerConfig(PretrainedConfig):
    def __init__(
        self,
        input_type: Literal["timm", "huggingface"] | None = None,
        return_patch_features: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_type = input_type
        self.return_patch_features = return_patch_features


class CLYPNeckConfig(PretrainedConfig):
    def __init__(
        self,
        in_channels: int = 768,
        out_channels: int = 512,
        bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias


class CLYPLossConfig(PretrainedConfig):
    def __init__(
        self,
        learn_temperature: bool = True,
        init_temperature: float = 0.07,
        max_temperature: Optional[float] = None,
        min_temperature: Optional[float] = None,
        label_smoothing: float = 0.0,
        gather_with_grad: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.learn_temperature = learn_temperature
        self.init_temperature = init_temperature
        self.max_temperature = max_temperature
        self.min_temperature = min_temperature
        self.label_smoothing = label_smoothing
        self.gather_with_grad = gather_with_grad


def _to_diff_dict(c: PretrainedConfig) -> dict:
    """Function to override PretrainedConfig.to_diff_dict()

    NOTE
    ----
    In transformers==4.38.1,
    PretrainedConfig.__repr__ may not be able to show configs that has some sub-configs
    """
    d = c.to_diff_dict()
    if "transformers_version" in d:
        d.pop("transformers_version")
    return d


if __name__ == "__main__":
    conf = CLYPConfig.from_pretrained("config.json")
    print(conf)
