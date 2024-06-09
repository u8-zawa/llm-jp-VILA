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

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.models.clip.modeling_clip import CLIPOutput

from .configuration_clyp import CLYPConfig, CLYPLossConfig
from .model import InfoNCELoss, create_text_encoder, create_vision_encoder
from .model_rinna import RinnaCLIPModel  # noqa


@dataclass
class CLYPOutput(CLIPOutput):
    ...


class CLYPPreTrainedModel(PreTrainedModel):
    config_class = CLYPConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_weights(self, module: Any) -> None:
        pass


class CLYPModel(CLYPPreTrainedModel):
    def __init__(self, config: CLYPConfig):
        super().__init__(config)
        self.vision_encoder = create_vision_encoder(config.vision_encoder_config)
        self.text_encoder = create_text_encoder(config.text_encoder_config)
        self.initialize_clip(
            learn_temperature=config.learn_temperature,
            temperature_init=config.temperature_init,
            temperature_min=config.temperature_min,
            temperature_max=config.temperature_max,
            itc_loss_config=config.itc_loss_config,
        )

    def initialize_clip(
        self,
        learn_temperature: Optional[bool] = None,
        temperature_init: Optional[float] = None,
        temperature_min: Optional[float] = None,
        temperature_max: Optional[float] = None,
        itc_loss_config: Optional[CLYPLossConfig] = None,
    ) -> None:
        # create contrastive loss function
        if itc_loss_config:
            raise NotImplementedError
        else:
            assert learn_temperature is not None
            assert temperature_init is not None
            self.itc_loss_fn = InfoNCELoss(
                learn_temperature=learn_temperature,
                init_temperature=temperature_init,
                max_temperature=temperature_max,
                min_temperature=temperature_min,
                gather_with_grad=True,
            )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> tuple | CLYPOutput:
        image_feats = self.vision_encoder(pixel_values)
        text_feats = self.text_encoder(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        )

        loss = None
        if return_loss:
            loss = self.itc_loss_fn(image_feats, text_feats)

        image_embeds = F.normalize(image_feats, dim=-1)
        text_embeds = F.normalize(text_feats, dim=-1)

        sim_i2t = image_embeds @ text_embeds.T
        sim_t2i = text_embeds @ image_embeds.T

        logits_per_image = sim_i2t / self.itc_loss_fn.temperature
        logits_per_text = sim_t2i / self.itc_loss_fn.temperature

        if not return_dict:
            if loss is None:
                return (logits_per_image, logits_per_text, text_embeds, image_embeds)
            return (loss, logits_per_image, logits_per_text, text_embeds, image_embeds)

        # TODO:
        #   - Support vision_model_output and text_model_output
        #   - Improve type: torch.Tensor -> torch.FloatTensor
        return CLYPOutput(
            loss=loss,
            logits_per_image=logits_per_image,  # type: ignore
            logits_per_text=logits_per_text,  # type: ignore
            text_embeds=text_embeds,  # type: ignore
            image_embeds=image_embeds,  # type: ignore
        )

    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        text_feats = self.text_encoder(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        )
        return text_feats

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        image_feats = self.vision_encoder(pixel_values)
        return image_feats


if __name__ == "__main__":
    model = CLYPModel.from_pretrained(".")
