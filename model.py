import torch.nn as nn
import torch.optim as optim
from transformers import RobertaModel, AutoTokenizer

class PhanLoaiTucTiu(nn.Module):
    
    def __init__(self, n_classes):
        super(PhanLoaiTucTiu, self).__init__()
        self.roberta = RobertaModel.from_pretrained('vinai/phobert-base')
        self.drop = nn.Dropout(p = 0.3)
        self.out = nn.Linear(self.roberta.config.hidden_size, n_classes)

    def forward(self, input_ids, token_type_ids, attention_mask):
        output_roberta = self.roberta(
            input_ids = input_ids,
            token_type_ids  = token_type_ids,
            attention_mask = attention_mask
        )

        pooled_output = output_roberta.pooler_output

        output = self.drop(pooled_output)

        return self.out(output)


