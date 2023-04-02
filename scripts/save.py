import torch
from transformers import AutoModel, PreTrainedModel

from transformer_ext.models import SampleConfig, SampleModel

model_dir: str = "model"
config: SampleConfig = SampleConfig(
    vocab_size=1000,
    hidden_size=128,
)
print(config)

model: SampleModel = SampleModel(config)
model.save_pretrained("model")

m: PreTrainedModel = AutoModel.from_pretrained("model")

input_ids: torch.LongTensor = torch.LongTensor([[0, 1, 2, 3]])
assert torch.allclose(model(input_ids), m(input_ids))
