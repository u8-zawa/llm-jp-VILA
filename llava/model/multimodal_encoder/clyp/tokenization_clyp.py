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

from typing import Optional

import torch
from transformers import BatchEncoding, PreTrainedTokenizer, T5Tokenizer
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)


class CLYPTokenizer(PreTrainedTokenizer):
    """CLYPTokenizer based on rinna/japanese-roberta-base

    This tokenizer is registered as a custom tokenizer to manually add CLS token to each text.
    """

    def __init__(self, max_length: int, padding: str, truncation: bool, **kwargs):
        # tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-roberta-base")
        self.tokenizer.do_lower_case = True

        super().__init__(
            max_length=max_length, padding=padding, truncation=truncation, **kwargs
        )
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def get_vocab(self) -> dict[str, int]:
        return self.tokenizer.get_vocab()

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> tuple[str]:
        return self.tokenizer.save_vocabulary(
            save_directory, filename_prefix=filename_prefix
        )

    def _tokenize(self, text, **kwargs):
        return self.tokenizer._tokenize(text, **kwargs)

    def _convert_token_to_id(self, token):
        return self.tokenizer._convert_token_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self.tokenizer._convert_id_to_token(index)

    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput],
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy | None = None,
        truncation: bool | str | TruncationStrategy | None = None,
        max_length: Optional[int] = None,
        **kwargs,
    ):
        if max_length is None:
            max_length = self.max_length
        if padding is None:
            padding = self.padding
        if truncation is None:
            truncation = self.truncation

        if add_special_tokens:
            max_length = max_length - 1

        if not isinstance(text, list):
            # TODO: Review
            text = [text]

        out = self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            add_special_tokens=False,
            **kwargs,
        )

        if add_special_tokens:
            input_ids = [
                [self.tokenizer.cls_token_id] + ids for ids in out["input_ids"]
            ]
            attention_mask = [[1] + am for am in out["attention_mask"]]
            position_ids = [list(range(0, len(input_ids[0])))] * len(input_ids)
        else:
            input_ids = out["input_ids"]
            attention_mask = out["attention_mask"]
            position_ids = [list(range(0, len(input_ids[0])))] * len(input_ids)

        # tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        position_ids = torch.tensor(position_ids, dtype=torch.long)

        # retrn
        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        return BatchEncoding(data=data)
