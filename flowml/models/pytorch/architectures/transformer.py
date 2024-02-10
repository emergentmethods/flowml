
import torch.nn as nn
from flowml.models.pytorch.utils import PositionalEncoding

"""
The architecture is based on the paper “Attention Is All You Need”.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Lukasz Kaiser, and Illia Polosukhin. 2017.
"""


class TransformerPositionalEncoding(nn.Module):
    def __init__(self,
                 input_features=7,
                 output_features=7,
                 num_layers=2,
                 dropout=0.1,
                 hidden_dim=1024,
                 lookback=10,
                 nhead=8):
        super(TransformerPositionalEncoding, self).__init__()
        self.dim_val = input_features - (input_features % nhead)

        # small dimension change to ensure input is divisible by nhead
        self.input_net = nn.Sequential(
            nn.Linear(input_features, self.dim_val)
        )

        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(
            d_model=self.dim_val, max_len=self.dim_val)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim_val, dim_feedforward=self.dim_val, nhead=nhead,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.output_net = nn.Sequential(
            nn.Linear(self.dim_val*lookback, int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim), int(hidden_dim/2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim/2), int(hidden_dim/4)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim/4), output_features)
        )

    def forward(self, input, mask=None, add_positional_encoding=True):
        """
        Args:
            x: Input features of shape [Batch, SeqLen, input_dim]
            mask: Mask to apply on the attention outputs (optional)
            add_positional_encoding: If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        x = self.input_net(input)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)
        # transform to a 1D vector space before decoding
        x = x.reshape(-1, 1, input.shape[1]*x.shape[-1])
        output = self.output_net(x)

        return output


# class TransformerTraditional(nn.Module):
#     def __init__(self, feature_size=7, output_features=7, num_layers=2, dropout=0.1, dim_val=1024,
#                  input_dropout=0.1, lookback=10):
#         super(TransformerTraditional, self).__init__()
#         self.input_net = nn.Sequential(
#             nn.Dropout(input_dropout), nn.Linear(feature_size, dim_val)
#         )
#         # Positional encoding for sequences
#         self.positional_encoding = PositionalEncoding(d_model=dim_val, max_len=dim_val)
#         self.encoder_layer = nn.TransformerEncoderLayer(
#             d_model=dim_val, nhead=8, dropout=dropout, batch_first=True)
#         self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
#         self.output_layer = nn.Linear(dim_val, output)

#     def forward(self, x, lookback, mask=None, add_positional_encoding=True):
#         """
#         Args:
#             x: Input features of shape [Batch, SeqLen, input_dim]
#             mask: Mask to apply on the attention outputs (optional)
#             add_positional_encoding: If True, we add the positional encoding to the input.
#                                       Might not be desired for some tasks.
#         """
#         x = self.input_net(x)
#         if add_positional_encoding:
#             x = self.positional_encoding(x)
#         x = self.transformer(x, mask=mask)
#         x = self.output_layer(x)
#         return x


# class Transformer(nn.Module):
#     # d_model : number of features
#     def __init__(self, feature_size=7, output=7, num_layers=2, dropout=0, dim_val=1024):
#         super(Transformer, self).__init__()
#         # Creating the three linear layers needed for the model
#         self.encoder_input_layer = nn.Linear(
#             in_features=feature_size,
#             out_features=dim_val
#         )

#         self.linear_mapping = nn.Linear(
#             in_features=dim_val,
#             out_features=output
#         )

#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_val, nhead=8, dropout=dropout)
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
#         self.decoder = nn.Linear(feature_size, output)
#         self.init_weights()

#     def init_weights(self):
#         initrange = 0.1
#         self.decoder.bias.data.zero_()
#         self.decoder.weight.data.uniform_(-initrange, initrange)

#     def _generate_square_subsequent_mask(self, sz):
#         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask

#     def forward(self, src):  # , device):

#         src = self.encoder_input_layer(src)
#         # mask = self._generate_square_subsequent_mask(len(src)).to(device)
#         output = self.transformer_encoder(src)  # , mask)
#         output = self.linear_mapping(output)
#         # output = self.decoder(output)
#         return output
