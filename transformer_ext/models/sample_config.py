from transformers import PretrainedConfig


class SampleConfig(PretrainedConfig):
    model_type: str = "sample_model"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
