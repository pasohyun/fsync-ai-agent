import torch.nn as nn
from transformers import AutoModel


class RoBERTaMultiLabel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        try:
            self.roberta = AutoModel.from_pretrained(model_name, attn_implementation="sdpa")
        except (ValueError, NotImplementedError):
            self.roberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls = output.last_hidden_state[:, 0, :].to(self.classifier.weight.dtype)
        logits = self.classifier(self.dropout(cls))
        return logits  # shape: [batch, num_labels], raw logits
