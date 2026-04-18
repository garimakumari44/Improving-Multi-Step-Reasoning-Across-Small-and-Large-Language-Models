import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=10):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)