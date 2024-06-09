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

from typing import Literal, Optional

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from timm.data import (
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
)
from timm.data.transforms_factory import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import ImageInput, make_list_of_images
from transformers.utils import TensorType

NormalizationType = Literal["imagenet", "imagenet_inception", "openai_clip"]


class CLYPImageProcessor(BaseImageProcessor):
    def __init__(
        self,
        image_size: int = 224,
        normalization_type: NormalizationType = "imagenet",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.normalization_type: NormalizationType = normalization_type

    def preprocess(
        self,
        images: ImageInput | list[ImageInput],
        return_tensors: Optional[str | TensorType] = None,
        **kwargs,
    ) -> BatchFeature:
        images = make_list_of_images(images, expected_ndims=3)
        # TODO: Support train
        transforms = TestTransform(
            self.image_size, normalization_type=self.normalization_type
        )
        images = [transforms(image).numpy() for image in images]
        return BatchFeature(data={"pixel_values": images}, tensor_type=return_tensors)


class TrainTransform:
    def __init__(
        self,
        image_size: int,
        scale_range_min: float,
        scale_range_max: float,
        normalization_type: NormalizationType = "imagenet",
    ) -> None:
        """
        Args:
            image_size (int): output-image size.
            scale_range_min (float): minimum value of the scale to crop an input image.
            scale_range_max (float): maximum value of the scale to crop an input image.
            normalization_type (str): select mean and std for normalization (see get_mean_and_std).
        """
        scale = (scale_range_min, scale_range_max)
        mean_and_std = get_mean_and_std(normalization_type)

        self.transform = T.Compose(
            [
                T.RandomResizedCrop(
                    image_size, scale=scale, interpolation=T.InterpolationMode.BICUBIC
                ),
                _convert_to_rgb,
                T.ToTensor(),
                T.Normalize(**mean_and_std),
            ]
        )

    def __call__(self, img):
        return self.transform(img)


class TestTransform:
    def __init__(
        self, image_size: int, normalization_type: NormalizationType = "imagenet"
    ) -> None:
        """
        Args:
            image_size (int): output-image size.
            normalization_type (str): select mean and std for normalization (see get_mean_and_std).
        """
        mean_and_std = get_mean_and_std(normalization_type)

        self.transform = T.Compose(
            [
                ResizeMaxSize(image_size, fill=0),
                T.CenterCrop(image_size),
                _convert_to_rgb,
                T.ToTensor(),
                T.Normalize(**mean_and_std),
            ]
        )

    def __call__(self, img):
        return self.transform(img)


class SmallestMaxSize(T.Resize):
    """Resize shorter side of an input image.

    The shorter side of an input image is resized to the max_size.
    Note that an large part of the input image is discarded when an aspect-ratio value of the input image is extremely small or large.
    """

    def __init__(self, max_size: int, **kwargs):
        super().__init__(max_size, **kwargs)

    @staticmethod
    def target_size(w: int, h: int, size: int) -> tuple[int, int]:
        if h < w:
            w, h = int(size * w / h), size
        else:
            w, h = size, int(size * h / w)
        return (h, w)

    def __call__(self, img):
        size = self.size
        assert isinstance(size, int)
        w, h = img.size
        target_size = self.target_size(w, h, size)
        return F.resize(img, list(target_size), self.interpolation)


class ResizeMaxSize(nn.Module):
    """Resize longer side of an input image.

    The longer side of an input image is resized to the max_size.
    Note that an large part of the output image is padded when an aspect-ration value of the input image is extremely small or large.
    Adapted from https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transform.py
    """

    def __init__(
        self,
        max_size: int,
        interpolation: T.InterpolationMode = T.InterpolationMode.BICUBIC,
        fn: str = "max",
        fill: int = 0,
    ):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == "min" else min
        self.fill = fill

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[:2]
        else:
            width, height = img.size
        scale = self.max_size / float(max(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = F.resize(img, new_size, self.interpolation)  # type: ignore
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = F.pad(
                img,
                padding=[
                    pad_w // 2,
                    pad_h // 2,
                    pad_w - pad_w // 2,
                    pad_h - pad_h // 2,
                ],
                fill=self.fill,
            )
        return img


def get_mean_and_std(normalization_type: NormalizationType) -> dict:
    """Return mean and std tensors for T.Normalize()
    NOTE:
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
        IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
        IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
        OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
        OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
    """
    if normalization_type == "imagenet":
        return {
            "mean": torch.tensor(IMAGENET_DEFAULT_MEAN),
            "std": torch.tensor(IMAGENET_DEFAULT_STD),
        }
    elif normalization_type == "imagenet_inception":
        return {
            "mean": torch.tensor(IMAGENET_INCEPTION_MEAN),
            "std": torch.tensor(IMAGENET_INCEPTION_STD),
        }
    elif normalization_type == "openai_clip":
        return {
            "mean": torch.tensor(OPENAI_CLIP_MEAN),
            "std": torch.tensor(OPENAI_CLIP_STD),
        }
    else:
        raise ValueError(normalization_type)


def _convert_to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")
