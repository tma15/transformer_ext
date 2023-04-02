import torch.nn as nn
from torch import FloatTensor
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from typing import Type

from .sample_config import SampleConfig


def register_to_hf_auto(model_class: Type[PreTrainedModel]):
    config_class: Type[PretrainedConfig] = model_class.config_class
    AutoConfig.register(model_class.config_class.model_type, config_class)
    AutoModel.register(config_class, model_class)


@register_to_hf_auto
class SampleModel(PreTrainedModel):
    config_class: PretrainedConfig = SampleConfig

    def __init__(self, config):
        super().__init__(config)

        self.fc1: nn.Module = nn.Linear(config.hidden_size, 2 * config.hidden_size)
        self.fc2: nn.Module = nn.Linear(2 * config.hidden_size, config.hidden_size)

    def forward(self, tensor: FloatTensor) -> FloatTensor:
        return self.fc2(self.fc1(tensor))
