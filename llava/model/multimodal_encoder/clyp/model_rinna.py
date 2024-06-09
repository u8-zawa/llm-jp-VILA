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

# Almost copied from https://github.com/rinnakk/japanese-clip/blob/master/src/japanese_clip/clip/modeling_clip.py
# This code is distributed under the Apache License 2.0.
from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.distributed.nn
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.models.clip import (
    CLIPVisionConfig,
    CLIPVisionModel,
)
from transformers.models.clip.modeling_clip import CLIPOutput
from transformers.utils import logging

logger = logging.get_logger(__name__)


# Copied from transformers.models.clip.modeling_clip.contrastive_loss
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits, torch.arange(len(logits), device=logits.device)
    )


# Copied from transformers.models.clip.modeling_clip.clip_loss
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.T)
    return (caption_loss + image_loss) / 2.0


class RinnaCLIPConfig(PretrainedConfig):
    model_type = "clip"
    is_composition = True

    def __init__(self, projection_dim=512, logit_scale_init_value=2.6592, **kwargs):
        super().__init__(**kwargs)

        if "vision_config" not in kwargs:
            raise ValueError("`vision_config` can not be `None`.")

        if "text_config" not in kwargs:
            raise ValueError("`text_config` can not be `None`.")

        vision_config = kwargs.pop("vision_config")
        text_config = kwargs.pop("text_config")

        vision_model_type = vision_config.pop("model_type")
        text_model_type = text_config.pop("model_type")

        if vision_model_type == "clip":
            self.vision_config = AutoConfig.for_model(
                vision_model_type, **vision_config
            ).vision_config
        elif vision_model_type == "clip_vision_model":
            self.vision_config = CLIPVisionConfig(**vision_config)
        else:
            self.vision_config = AutoConfig.for_model(
                vision_model_type, **vision_config
            )

        self.text_config = AutoConfig.for_model(text_model_type, **text_config)

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value

    @classmethod
    def from_vision_text_configs(
        cls, vision_config: PretrainedConfig, text_config: PretrainedConfig, **kwargs
    ):
        r"""
        Instantiate a [`VisionTextDualEncoderConfig`] (or a derived class) from text model configuration and vision
        model configuration.

        Returns:
            [`VisionTextDualEncoderConfig`]: An instance of a configuration object
        """

        return cls(
            vision_config=vision_config.to_dict(),
            text_config=text_config.to_dict(),
            **kwargs,
        )

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output


class RinnaCLIPModel(PreTrainedModel):
    config_class = RinnaCLIPConfig
    base_model_prefix = "clip"

    def __init__(
        self,
        config: Optional[RinnaCLIPConfig] = None,
        vision_model: Optional[PreTrainedModel] = None,
        text_model: Optional[PreTrainedModel] = None,
    ):
        if config is None and (vision_model is None or text_model is None):
            raise ValueError(
                "Either a configuration or an vision and a text model has to be provided"
            )

        if config is None:
            config = RinnaCLIPConfig.from_vision_text_configs(
                vision_model.config,
                text_model.config,  # type: ignore[union-attr]
            )
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(
                    f"config: {config} has to be of type {self.config_class}"
                )

        # initialize with config
        super().__init__(config)

        if vision_model is None:
            if isinstance(config.vision_config, CLIPVisionConfig):
                vision_model = CLIPVisionModel(
                    config.vision_config, add_pooling_layer=False
                )
            else:
                vision_model = AutoModel.from_config(
                    config.vision_config, add_pooling_layer=False
                )

        if text_model is None:
            text_model = AutoModel.from_config(
                config.text_config, add_pooling_layer=False
            )

        self.vision_model = vision_model
        self.text_model = text_model

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.vision_model.config = self.config.vision_config
        self.text_model.config = self.config.text_config

        self.vision_embed_dim = config.vision_config.hidden_size
        self.text_embed_dim = config.text_config.hidden_size
        self.projection_dim = config.projection_dim

        self.visual_projection = nn.Linear(
            self.vision_embed_dim, self.projection_dim, bias=False
        )
        self.text_projection = nn.Linear(
            self.text_embed_dim, self.projection_dim, bias=False
        )
        self.logit_scale = nn.Parameter(
            torch.ones([]) * self.config.logit_scale_init_value
        )

    def get_text_features(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        out=False,
    ):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = text_outputs.last_hidden_state[:, 0, :]
        text_features = self.text_projection(pooled_output)
        if out:
            return text_features, text_outputs
        return text_features

    def get_image_features(
        self,
        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs.last_hidden_state[:, 0, :]
        image_features = self.visual_projection(pooled_output)

        return image_features

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        return_loss=None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs.last_hidden_state[:, 0, :]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs.last_hidden_state[:, 0, :]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        # logit_scale = self.logit_scale
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (
                logits_per_image,
                logits_per_text,
                text_embeds,
                image_embeds,
                text_outputs,
                vision_outputs,
            )
            return ((loss,) + output) if loss is not None else output

        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported
        # for composite models
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_vision_text_pretrained(
        cls,
        vision_model_name_or_path: Optional[str] = None,
        text_model_name_or_path: Optional[str] = None,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:
        kwargs_vision = {
            argument[len("vision_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("vision_")
        }

        kwargs_text = {
            argument[len("text_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("text_")
        }

        # remove vision, text kwargs from kwargs
        for key in kwargs_vision.keys():
            del kwargs["vision_" + key]
        for key in kwargs_text.keys():
            del kwargs["text_" + key]

        # Load and initialize the vision and text model
        vision_model = kwargs_vision.pop("model", None)
        if vision_model is None:
            if vision_model_name_or_path is None:
                raise ValueError(
                    "If `vision_model` is not defined as an argument, a `vision_model_name_or_path` has to be defined"
                )

            if "config" not in kwargs_vision:
                vision_config = AutoConfig.from_pretrained(vision_model_name_or_path)

            if vision_config.model_type == "clip":
                kwargs_vision["config"] = vision_config.vision_config
                vision_model = CLIPVisionModel.from_pretrained(
                    vision_model_name_or_path,
                    add_pooling_layer=False,
                    *model_args,
                    **kwargs_vision,
                )
                # TODO: Should we use the pre-trained projection as well ?
            else:
                kwargs_vision["config"] = vision_config
                vision_model = AutoModel.from_pretrained(
                    vision_model_name_or_path,
                    add_pooling_layer=False,
                    *model_args,
                    **kwargs_vision,
                )

        text_model = kwargs_text.pop("model", None)
        if text_model is None:
            if text_model_name_or_path is None:
                raise ValueError(
                    "If `text_model` is not defined as an argument, a `text_model_name_or_path` has to be defined"
                )

            if "config" not in kwargs_text:
                text_config = AutoConfig.from_pretrained(text_model_name_or_path)
                kwargs_text["config"] = text_config

            text_model = AutoModel.from_pretrained(
                text_model_name_or_path,
                add_pooling_layer=False,
                *model_args,
                **kwargs_text,
            )

        # instantiate config with corresponding kwargs
        config = RinnaCLIPConfig.from_vision_text_configs(
            vision_model.config, text_model.config, **kwargs
        )

        # init model
        model = cls(config=config, vision_model=vision_model, text_model=text_model)

        # the projection layers are always newly initialized when loading the model
        # using pre-trained vision and text model.
        # logger.warning(
        #    "The projection layer and logit scale weights `['visual_projection.weight', 'text_projection.weight', 'logit_scale']` "
        #    "are newly initialized. You should probably TRAIN this model on a down-stream task "
        #    "to be able to use it for predictions and inference."
        # )

        return model
