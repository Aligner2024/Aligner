# Copyright 2023-2024 PKU-Alignment Team. All Rights Reserved.
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
# ==============================================================================
# This file is modified from:
# https://github.com/PKU-Alignment/safe_rlhf/datasets/raw/alpaca.py
# ==============================================================================
"""Stanford Alpaca dataset for supervised instruction fine-tuning."""

from __future__ import annotations

from datasets import load_dataset
from training.datasets.base import RawDataset, RawSample
import json

__all__ = ['AlpacaDataset','AlpacaJSONDataset']


class AlpacaDataset(RawDataset):
    NAME: str = 'alpaca'
    ALIASES: tuple[str, ...] = ('stanford-alpaca',)

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset(path or 'tatsu-lab/alpaca', split='train')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        input = (  # pylint: disable=redefined-builtin
            ' '.join((data['instruction'], data['input'])) if data['input'] else data['instruction']
        )
        answer = data['output']
        return RawSample(input=input, answer=answer)

    def __len__(self) -> int:
        return len(self.data)

class AlpacaJSONDataset(RawDataset):
    NAME: str = 'alpaca-json'

    def __init__(self, path) -> None:  # noqa: ANN001
        self.path = path
        with open(self.path, encoding='utf-8') as f:
            self.data = json.load(f)

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        input = (  # pylint: disable=redefined-builtin
            ' '.join((data['instruction'], data['input'])) if data['input'] else data['instruction']
        )
        answer = data['output']
        return RawSample(input=input, answer=answer)

    def __len__(self) -> int:
        return len(self.data)