import torch
import torch.nn as nn


class CreditScoringMLP(nn.Module):
    """
        Fully Connected MLP for credit default prediction.

        Architecture:
            Input  (n_features)
            Dense(32) + ReLU + BatchNorm + Dropout(0.3)
            Dense(16) + ReLU + BatchNorm + Dropout(0.2)
            Dense(8)  + ReLU
            Output(1) + Sigmoid
            
    """
    def __init__(self, n_features: int, DROPOUT_1: float = 0.3, DROPOUT_2: float = 0.2):
        super().__init__()

        self.network = nn.Sequential(
            # Block 1
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(DROPOUT_1),

            # Block 2
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(DROPOUT_2),

            # Block 3
            nn.Linear(16, 8),
            nn.ReLU(),
            
            # Block 4
            nn.Linear(8, 2),
            nn.ReLU(),

            # Output
            nn.Linear(2, 1),
            nn.Sigmoid(),
        )

        # Weight initialization (Xavier for stable gradients)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)







