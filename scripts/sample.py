import torch
from transformers import AutoModel, PreTrainedModel

from transformer_ext.models import SampleConfig, SampleModel

model_dir: str = "model"
config: SampleConfig = SampleConfig(hidden_size=128)
print(config)

model: SampleModel = SampleModel(config)
model.save_pretrained("model")

m: PreTrainedModel = AutoModel.from_pretrained("model")

assert torch.allclose(model.fc1.weight, m.fc1.weight)
assert torch.allclose(model.fc2.weight, m.fc2.weight)
