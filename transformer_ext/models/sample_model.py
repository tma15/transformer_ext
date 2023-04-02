import torch.nn as nn
from torch import FloatTensor
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel

from .sample_config import SampleConfig


class SampleModel(PreTrainedModel):
    config_class: PretrainedConfig = SampleConfig

    def __init__(self, config):
        super().__init__(config)

        self.fc1: nn.Module = nn.Linear(config.hidden_size, 2 * config.hidden_size)
        self.fc2: nn.Module = nn.Linear(2 * config.hidden_size, config.hidden_size)

    def forward(self, tensor: FloatTensor) -> FloatTensor:
        return self.fc2(self.fc1(tensor))


AutoConfig.register("sample_model", SampleConfig)
AutoModel.register(SampleConfig, SampleModel)
