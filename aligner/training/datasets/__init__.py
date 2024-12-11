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
# https://github.com/PKU-Alignment/safe_rlhf/datasets/__init__.py
# ==============================================================================
"""Dataset classes."""

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import Dataset

from training.datasets import raw
from training.datasets.base import (
    CollatorBase,
    RawDataset,
    RawSample,
    TokenizedDataset,
    parse_dataset,
)
from training.datasets.preference import (
    PreferenceBatch,
    PreferenceCollator,
    PreferenceDataset,
    PreferenceSample,
)
from training.datasets.prompt_only import (
    PromptOnlyBatch,
    PromptOnlyCollator,
    PromptOnlyDataset,
    PromptOnlySample,
)
from training.datasets.raw import *  # noqa: F403
from training.datasets.safety_preference import (
    SafetyPreferenceBatch,
    SafetyPreferenceCollator,
    SafetyPreferenceDataset,
    SafetyPreferenceSample,
)
from training.datasets.supervised import (
    SupervisedBatch,
    SupervisedCollator,
    SupervisedDataset,
    SupervisedSample,
)


__all__ = [
    'DummyDataset',
    'parse_dataset',
    'RawDataset',
    'RawSample',
    'TokenizedDataset',
    'CollatorBase',
    'PreferenceDataset',
    'PreferenceSample',
    'PreferenceBatch',
    'PreferenceCollator',
    'PromptOnlyDataset',
    'PromptOnlyCollator',
    'PromptOnlySample',
    'PromptOnlyBatch',
    'SafetyPreferenceDataset',
    'SafetyPreferenceCollator',
    'SafetyPreferenceSample',
    'SafetyPreferenceBatch',
    'SupervisedDataset',
    'SupervisedCollator',
    'SupervisedSample',
    'SupervisedBatch',
    *raw.__all__,
]


class DummyDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {}
