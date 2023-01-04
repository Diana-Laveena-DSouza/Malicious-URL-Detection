import torch.nn as nn
import torch
import math

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, vocab_size=256, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    
    def __init__(self):
        super(TransformerModel, self).__init__()
        # embedding layer
        self.embedding = nn.Embedding(256, 256)
        # positional encoding layer
        self.positional_encoding = PositionalEncoding(256, 256, 0.1)
        
        # encoder  layers
        enc_layer = nn.TransformerEncoderLayer(d_model = 256, nhead = 4, dim_feedforward = 128, dropout = 0.01)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers = 6)
        
        # final dense layer
        self.classifier = nn.Linear(256, 2)
        self.d_model = 256
                
    def forward(self, token_id):
        x = self.embedding(token_id.to(device)) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        pred = self.classifier(x)
        return pred