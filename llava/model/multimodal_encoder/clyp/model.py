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

import logging
from typing import Optional, Union

import timm
import torch
import torch.distributed as dist
import torch.distributed.nn
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import SwinTransformer as TimmSwinTransformer
from transformers import PreTrainedModel
from transformers.utils.logging import get_logger

from .configuration_clyp import (
    CLYPTextBackboneConfig,
    CLYPTextEncoderConfig,
    CLYPVisionBackboneConfig,
    CLYPVisionEncoderConfig,
)
from .model_rinna import RinnaCLIPConfig, RinnaCLIPModel

DEFAULT_LOGGER = get_logger(__name__)


class VisionEncoder(nn.Module):
    """Vision encoder to extract image feateurs.

    Pooler and neck are optional.
    Instead of defining pooler and neck in VisionEncoder, you can define them in algorithm classes.

    Attributes:
        backbone (nn.Module): backbone loaded from timm, huggingface or registry.
        pooler (nn.Module): module to extract image-level features.
        neck (nn.Module): module to adjust feature dimensions.
    """

    def __init__(
        self,
        backbone: nn.Module,
        pooler: Optional[nn.Module] = None,
        neck: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pooler = pooler
        self.neck = neck

    def forward(self, imgs: torch.Tensor):
        """A method to extract image features.

        Args:
            imgs (torch.Tensor): shape=(batch_size, channels, height, width).

        Returns:
            out (torch.Tensor): the output shape changes depending on pooler, and the following shapes are usually expected.
                - output only image-level features like CLIP: shape=(batch_size, embed_dim)
                - output image-level and local patch features like BLIP2: shape=(batch_size, embed_dim, length)
        """
        out = self.backbone(imgs)  # Shape=(batch_size, channels, height, width)
        if self.pooler:
            out = self.pooler(out)
        if self.neck:
            out = self.neck(out)
        return out


class SwinTransformerPerm(nn.Module):
    """Wrapper for SwinTransformer in timm.

    This wrapper changes the output shape to (batch_size, channels, height, width).
    The original shape of timm SwinTransformer is (batch_size, height, width, channels).
    """

    def __init__(self, swin: nn.Module) -> None:
        super().__init__()
        self.swin = swin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.swin(x)
        out = out.permute(0, 3, 1, 2)
        return out


def load_from_timm(
    config: CLYPVisionBackboneConfig,
    use_gradient_checkpointing: bool = False,
    path_weights: Optional[str] = None,
    logger: logging.Logger = DEFAULT_LOGGER,
):
    """Create a backbone using a method: timm.create_model.

    Args:
        config (TimmBackboneConfig): config fed to timm.create_model.
        use_gradient_checkpointing (bool): True if use gradient checkpointing.
        path_weights (str): path to weights for backbone initialization.
    """
    # backbone
    assert config is not None
    backbone = timm.create_model(
        model_name=config.model_name,
        pretrained=config.pretrained,
        **config.extra_kwargs,
    )
    backbone.reset_classifier(0, "")

    logger.info(
        f"    - load from timm: model_name={config.model_name}, pretrained={config.pretrained}"
    )

    # gradient checkpointing
    backbone.set_grad_checkpointing(enable=use_gradient_checkpointing)
    if use_gradient_checkpointing:
        logger.info("    - gradient checkpointing is enebled.")

    # init weights
    if path_weights:
        state_dict = torch.load(path_weights, map_location="cpu")
        checks = backbone.load_state_dict(state_dict, strict=False)
        logger.info(f"    - load weights from {path_weights}")
        logger.info(f"    - state dict checks: {checks}")

    # swin
    if isinstance(backbone, TimmSwinTransformer):
        backbone = SwinTransformerPerm(backbone)
    return backbone


def create_vision_encoder(
    config: CLYPVisionEncoderConfig, logger: logging.Logger = DEFAULT_LOGGER
) -> VisionEncoder:
    assert config.pooler_config.input_type
    backbone = load_from_timm(config.backbone_config, logger=logger)
    pooler = CLSTokenPooling(
        config.pooler_config.input_type, config.pooler_config.return_patch_features
    )
    neck = Linear(
        config.neck_config.in_channels,
        config.neck_config.out_channels,
        config.neck_config.bias,
    )
    return VisionEncoder(backbone, pooler=pooler, neck=neck)


class TextEncoder(nn.Module):
    """Text encoder to extract text features.

    Pooler and neck are optional.
    Instead of defining pooler and neck in TextEncoder, you can define them in algorithm classes.

    Attributes:
        backbone (nn.Module): backbone loaded from timm, huggingface or registry.
        pooler (nn.Module): module to extract image-level features.
        neck (nn.Module): module to adjust feature dimensions.

    """

    def __init__(
        self,
        backbone: nn.Module,
        pooler: Optional[nn.Module] = None,
        neck: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pooler = pooler
        self.neck = neck

    def forward(self, inputs: dict) -> torch.Tensor:
        """A method to extract text features.

        Args:
            inputs (dict): basic keys are shown below:
                - input_ids (torch.Tensor)
                - attention_mask (Optional[torch.Tensor])
                - position_ids (Optional[torch.Tensor])
                - token_type_ids (Optional[torch.Tensor])
                - output_attentions Optional[bool]
                - output_hidden_states Optional[bool]

        Returns:
            out (torch.Tensor): the output shape changes depending on pooler, and the following shapes are usually expected.
                - output only class token like CLIP: shape=(batch_size, embed_dim)
                - output all token features like BLIP2: shape=(batch_size, embed_dim, length)
        """
        out = self.backbone(**inputs)
        if self.pooler:
            out = self.pooler(out)
        if self.neck:
            out = self.neck(out)
        return out


class TextBackboneModelWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model.text_model

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        return out

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        if enabled:
            self.model.gradient_checkpointing_enable()


def load_from_huggingface(
    config: CLYPTextBackboneConfig,
    use_gradient_checkpointing: bool = False,
    path_weights: Optional[str] = None,
    logger: logging.Logger = DEFAULT_LOGGER,
) -> nn.Module:
    """Load a backbone from huggingface.

    Args:
        config (HuggingfaceBackboneConfig): config fed to AutoModel.from_pretrained.
        use_gradient_checkpointing (bool): True if use gradient checkpointing.
        path_weights (str): path to weights for backbone initialization.
    """

    # NOTE:
    # Initialize Rinna CLIP without pretrained weights here,
    # because CLYP model loads its whole weights afterward
    auto_config = RinnaCLIPConfig.from_pretrained(config.model_name)
    backbone = RinnaCLIPModel(auto_config)

    logger.info(f"    - load from huggingface: model_name={config.model_name}")

    # gradient checkpointing
    if isinstance(backbone, PreTrainedModel):
        if use_gradient_checkpointing:
            backbone.gradient_checkpointing_enable()
            logger.info("    - gradient checkpointing is enabled")
    else:
        raise NotImplementedError()

    # init weights
    if path_weights:
        raise NotImplementedError()
    return backbone


def create_text_encoder(
    config: CLYPTextEncoderConfig, logger: logging.Logger = DEFAULT_LOGGER
) -> TextEncoder:
    assert config.pooler_config.input_type
    backbone = TextBackboneModelWrapper(
        load_from_huggingface(config.backbone_config, logger=logger)
    )
    pooler = CLSTokenPooling(
        config.pooler_config.input_type, config.pooler_config.return_patch_features
    )
    neck = Linear(
        config.neck_config.in_channels,
        config.neck_config.out_channels,
        bias=config.neck_config.bias,
    )
    return TextEncoder(backbone, pooler=pooler, neck=neck)


class Linear(nn.Module):
    """Linear layer."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool) -> None:
        """
        Args:
            in_channels (int): input feature dimension.
            out_channels (out): output feature dimension.
            bias (bool): True if use bias in nn.Linear.
        """
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape=(batch_size, ..., in_channels).

        Returns:
            out (torch.Tensor): shape=(batch_size, ..., out_channels).
        """
        out = self.linear(x)
        return out


class CLSTokenPooling(nn.Module):
    """A module to extract class token."""

    def __init__(self, input_type: str, return_patch_features: bool) -> None:
        """
        Args:
            input_type (str): timm or huggingface.
                - If input_type is timm, x[:, 0] is extracted as a class token.
                - If input_type is huggingface, x.last_hidden_state[:,0] is extracted as a class token.
            return_patch_features (bool): True if output local features.
        """
        super().__init__()
        assert input_type in ["timm", "huggingface"]
        self.input_type = input_type
        self.return_patch_features = return_patch_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape=(batch_size, length, dim).

        Returns:
            out (torch.Tensor): shape=(batch_size, dim).
        """
        # tensor: shape=(batch_size, length, dim)
        if self.input_type == "timm":
            assert x.ndim == 3, "CLSTokenPooling: dimension of input tensor must be 3."
            if self.return_patch_features:
                return x
            else:
                return x[:, 0]

        # huggingface
        elif self.input_type == "huggingface":
            out = x.last_hidden_state
            if self.return_patch_features:
                return out
            else:
                return out[:, 0]


class InfoNCELoss(nn.Module):
    def __init__(
        self,
        learn_temperature: bool,
        init_temperature: float,
        max_temperature: Optional[float] = None,
        min_temperature: Optional[float] = None,
        label_smoothing: float = 0.0,
        gather_with_grad: bool = False,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.gather_with_grad = gather_with_grad

        # set temperature
        self.learn_temperature = learn_temperature
        self.temperature = torch.ones([]) * init_temperature
        if self.learn_temperature:
            self.temperature = nn.Parameter(self.temperature)
            self.max_temperature = max_temperature
            self.min_temperature = min_temperature

        # whether clip temperature or not
        self.require_temperature_clipping = self.learn_temperature and (
            self.max_temperature or self.min_temperature
        )

    def clip_temperature(self):
        if self.require_temperature_clipping:
            self.temperature.data = torch.clamp(
                self.temperature, self.min_temperature, self.max_temperature
            )

    def forward(
        self,
        image_feats: torch.Tensor,
        text_feats: torch.Tensor,
        return_similarity: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor]]:
        # gather image and text features
        image_feats_all = concat_all_gather(
            image_feats, with_grad=self.gather_with_grad
        )
        text_feats_all = concat_all_gather(text_feats, with_grad=self.gather_with_grad)

        # compute cosine similarity
        sim_i2t = image_to_text_similarity(
            image_feats=image_feats,
            text_feats=text_feats_all,
        )
        sim_t2i = text_to_image_similarity(
            text_feats=text_feats,
            image_feats=image_feats_all,
        )

        # logits, scaled cosine similarity
        logits_i2t = sim_i2t / self.temperature
        logits_t2i = sim_t2i / self.temperature

        # obtain targets
        rank = dist.get_rank()
        batch_size = image_feats.size(0)
        targets = torch.arange(batch_size) + batch_size * rank
        targets = targets.to(dtype=torch.long, device=image_feats.device)

        # calculate loss
        loss_i2t = F.cross_entropy(
            logits_i2t, targets, label_smoothing=self.label_smoothing
        )
        loss_t2i = F.cross_entropy(
            logits_t2i, targets, label_smoothing=self.label_smoothing
        )
        loss = (loss_i2t + loss_t2i) / 2.0

        if not return_similarity:
            return loss
        else:
            return loss, sim_i2t, sim_t2i


def image_to_text_similarity(
    image_feats: torch.Tensor, text_feats: torch.Tensor
) -> torch.Tensor:
    """
    Args:
        image_feats (torch.Tensor): shape=(num_imgs, embed_dim) or (num_imgs, num_query_tokens, embed_dim).
        text_feats (torch.Tensor): shape=(num_texts, embed_dim).

    Returns:
        sim_i2t (torch.Tensor): shape=(num_imgs, num_texts).
    """
    assert image_feats.ndim in [2, 3]
    assert text_feats.ndim == 2

    # normalize features
    image_feats = F.normalize(image_feats, dim=-1)
    text_feats = F.normalize(text_feats, dim=-1)

    if image_feats.ndim == 2:
        sim_i2t = image_feats @ text_feats.T
    else:
        # a query token with maximum cosine similarity is selected
        sim_i2t = torch.matmul(
            image_feats.unsqueeze(1), text_feats.unsqueeze(0).unsqueeze(-1)
        ).squeeze()  # shape=(num_imgs, num_texts, num_query_tokens)
        sim_i2t, _ = sim_i2t.max(dim=-1)  # shape=(num_imgs, num_texts)
    return sim_i2t


def text_to_image_similarity(text_feats: torch.Tensor, image_feats: torch.Tensor):
    """
    Args:
        text_feats (torch.Tensor): shape=(num_texts, embed_dim).
        image_feats (torch.Tensor): shape=(num_imgs, embed_dim) or (num_imgs, num_query_tokens, embed_dim).

    Returns:
        similarity_maxtrix (torch.Tensor): shape=(num_texts, num_imgs).
    """
    assert image_feats.ndim in [2, 3]
    assert text_feats.ndim == 2

    # normalize features
    image_feats = F.normalize(image_feats, dim=-1)
    text_feats = F.normalize(text_feats, dim=-1)

    if image_feats.ndim == 2:
        sim_t2i = text_feats @ image_feats.T
    else:
        # a query token with maximum cosine similarity is selected
        sim_t2i = torch.matmul(
            text_feats.unsqueeze(1).unsqueeze(1),
            image_feats.permute(0, 2, 1).unsqueeze(0),
        ).squeeze()
        sim_t2i, _ = sim_t2i.max(dim=-1)
    return sim_t2i


def concat_all_gather(tensor: torch.Tensor, with_grad: bool):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.

    Another implementation: https://github.com/salesforce/LAVIS/blob/main/lavis/models/base_model.py#L202-L237
    """
    if with_grad:
        output = torch.cat(torch.distributed.nn.all_gather(tensor), dim=0)
    else:
        tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(tensors_gather, tensor, async_op=False)
        output = torch.cat(tensors_gather, dim=0)
    return output
