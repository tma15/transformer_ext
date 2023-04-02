import torch.nn as nn
from torch import FloatTensor, LongTensor
from transformers import AutoModel, PretrainedConfig, PreTrainedModel
from typing import Type, Optional

from .sample_config import SampleConfig


def register_to_hf_auto_model(
    model_class: Type[PreTrainedModel],
) -> Type[PreTrainedModel]:
    config_class: Type[PretrainedConfig] = model_class.config_class
    AutoModel.register(config_class, model_class)
    return model_class


@register_to_hf_auto_model
class SampleModel(PreTrainedModel):
    config_class: PretrainedConfig = SampleConfig

    def __init__(self, config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.fc1: nn.Module = nn.Linear(config.hidden_size, 2 * config.hidden_size)
        self.fc2: nn.Module = nn.Linear(2 * config.hidden_size, config.hidden_size)

    def forward(
        self,
        input_ids: Optional[LongTensor] = None,
        attention_mask: Optional[LongTensor] = None,
        position_ids: Optional[LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):

        embeddings: FloatTensor = self.embeddings(input_ids)
        return self.fc2(self.fc1(embeddings))
