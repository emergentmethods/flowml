import torch


class Net(torch.nn.Module):
    def __init__(self, input_features=32, output_features=6, output_dim=6):
        super().__init__()
        self.output_dim = output_dim
        self.output_features = output_features
        self.hidden_features = 1024
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_features, self.hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_features, self.hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_features,
                            output_features * output_dim),
        )

    def forward(self, x):
        y = self.net(x)
        return y


class MLPBase(torch.nn.Module):
    def __init__(self, input_features, output_features, dropout=0.1, hidden_dim=256):
        super().__init__()

        self.input_net = torch.nn.Sequential(
            torch.nn.Linear(int(input_features), int(hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(int(hidden_dim), int(hidden_dim/2)),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(int(hidden_dim/2), int(hidden_dim/4)),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(int(hidden_dim/4), int(output_features))
        )

    def forward(self, x):
        y = self.input_net(x)
        # x = self.net(x)
        # y = self.output_layer(x)
        return y
