from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling

from .configuration_clyp import CLYPConfig, CLYPLossConfig
from .model import InfoNCELoss, create_vision_encoder
from .modeling_clyp import CLYPPreTrainedModel


class CLYPVisionModel(CLYPPreTrainedModel):
    def __init__(self, config: CLYPConfig):
        super().__init__(config)
        self.vision_encoder = create_vision_encoder(config.vision_encoder_config)
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
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> tuple | BaseModelOutputWithPooling:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        last_hidden_state = self.vision_encoder.backbone(pixel_values)

        # TODO:
        #   - Support return_dict=False
        if not return_dict:
            # return (loss, logits_per_image, logits_per_text, text_embeds, image_embeds)
            raise ValueError("Only support return_dict=True")

        # TODO:
        #   - return all hidden_states
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=None,
            hidden_states=tuple([last_hidden_state]),
            attentions=None,
        )
