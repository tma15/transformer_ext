from transformers import AutoModel, PreTrainedModel

import transformer_ext  # type: ignore

m: PreTrainedModel = AutoModel.from_pretrained("model")
print(m)
