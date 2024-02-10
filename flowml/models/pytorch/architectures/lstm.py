import torch.nn as nn

# define the LSTM model
class LSTMAutoRegressive(nn.Module):
    def __init__(self,
                 input_features,
                 hidden_dim,
                 output_features,
                 dropout=0.1,
                 lookback=8):
        super(LSTMAutoRegressive, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_net = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(input_features, hidden_dim)
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2,
                            batch_first=True, dropout=dropout)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim*lookback, output_features),
            nn.Tanh()
        )

    def forward(self, input, hn=None, cn=None):
        x = self.input_net(input)
        if hn is None or cn is None:
            x, (hn, cn) = self.lstm(x)
        else:
            x, (hn, cn) = self.lstm(x, (hn, cn))
        x = x.reshape(-1, 1, input.shape[1]*x.shape[-1])
        output = self.output_layer(x)
        # only interested in the last output of the LSTM
        # output = output[:, -1, :].unsqueeze(1)   # , (hn, cn)
        return output, hn, cn


# define the LSTM model
class LSTMDirect(nn.Module):
    def __init__(self,
                 input_features,
                 hidden_dim,
                 output_features,
                 dropout=0.1,
                 lookback=16,
                 num_layers=2):
        super(LSTMDirect, self).__init__()

        self.lstm = nn.LSTM(int(input_features), int(input_features), num_layers=num_layers,
                            batch_first=True, dropout=dropout)

        self.output_net = nn.Sequential(
            nn.Linear(int(input_features)*lookback, int(hidden_dim)),
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

    def forward(self, input, hn=None, cn=None):
        # x = self.input_net(input)
        if hn is None or cn is None:
            x, (hn, cn) = self.lstm(input)
        else:
            x, (hn, cn) = self.lstm(input, (hn, cn))
        # flatten to a 1D vector space before decoding
        x = x.reshape(-1, 1, input.shape[1]*x.shape[-1])
        output = self.output_net(x)

        return output, hn, cn
