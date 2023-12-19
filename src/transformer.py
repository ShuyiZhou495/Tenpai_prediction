import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
# class TransformerModel(nn.Transformer):
#     """Container module with an encoder, a recurrent or transformer module, and a decoder."""

#     def __init__(self, ntoken, nlen, ninp, nhead, nhid, nlayers, dropout=0.1):
#         super(TransformerModel, self).__init__(d_model=ninp, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers)
#         self.model_type = 'Transformer'
#         self.src_mask = None
#         self.pos_encoder = PositionalEncoding(ninp, dropout, max_len=nlen)
#         encoder_layers = TransformerEncoderLayer(52, nhead, nhid, dropout)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
#         # self.input_emb = nn.Embedding(ntoken, ninp)
#         self.ninp = ninp
#         self.decoder1 = nn.Linear(ninp, 256)
#         self.decoder2 = nn.Linear(256, 34)

#         self.init_weights()

#     def _generate_square_subsequent_mask(self, sz):
#         return torch.log(torch.tril(torch.ones(sz,sz)))

#     def init_weights(self):
#         initrange = 0.1
#         # nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
#         nn.init.zeros_(self.decoder1.bias)
#         nn.init.zeros_(self.decoder2.bias)
#         nn.init.uniform_(self.decoder1.weight, -initrange, initrange)
#         nn.init.uniform_(self.decoder2.weight, -initrange, initrange)

#     def forward(self, src, sizes, has_mask=True):
#         src_key_padding_mask = torch.arange(len(src), device="cuda:0").tile(len(sizes), 1)
#         tsize = sizes.tile(src.shape[0], 1).transpose(0, 1)
#         src_key_padding_mask[src_key_padding_mask < tsize] = 1
#         src_key_padding_mask[src_key_padding_mask >= tsize] = 0
#         src_key_padding_mask = torch.log(src_key_padding_mask)
#         src = src * math.sqrt(self.ninp)
#         src = self.pos_encoder(src)
#         output = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
#         output1 = F.relu(self.decoder1(output.mean(dim=0)))
#         return F.softmax(self.decoder2(output1), dim=-1)
    

class TransformerModel(nn.Module):
    def __init__(self, input_size, nlen, hidden_size, num_layers, num_heads, output_size, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # Linear layer to project one-hot encoded input to the desired hidden size
        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout, max_len=nlen)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size*4,
                dropout=dropout
            ),
            num_layers=num_layers
        )
        
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, sizes):
        embedded = self.embedding(x)
        src = self.pos_encoder(embedded)
        src_key_padding_mask = torch.arange(len(src), device="cuda:0").tile(len(sizes), 1)
        tsize = sizes.tile(src.shape[0], 1).transpose(0, 1)
        src_key_padding_mask[src_key_padding_mask < tsize] = 1
        src_key_padding_mask[src_key_padding_mask >= tsize] = 0
        src_key_padding_mask = torch.log(src_key_padding_mask)
        transformer_output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)

        # Assuming you want to pool over the sequence dimension
        pooled_output = transformer_output.mean(dim=0)

        x = self.fc1(pooled_output)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        output = self.softmax(x)

        return output    