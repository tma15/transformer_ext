from typing import Type

from transformers import AutoConfig, PretrainedConfig


def register_to_hf_auto_config(
    config_class: Type[PretrainedConfig],
) -> Type[PretrainedConfig]:
    AutoConfig.register(config_class.model_type, config_class)
    return config_class


@register_to_hf_auto_config
class SampleConfig(PretrainedConfig):
    model_type: str = "sample_model"

    def __init__(self, vocab_size: int = 1000, **kwargs):

        self.vocab_size: int = vocab_size
        super().__init__(**kwargs)
