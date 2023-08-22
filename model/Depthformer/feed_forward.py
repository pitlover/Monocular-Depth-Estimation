from typing import Optional
import torch
import torch.nn as nn


class FeedForwardBlock(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 feedforward_dim: Optional[int] = None,
                 drop_prob: float = 0.1,
                 act_layer=nn.GELU,
                 add_weight: float = 1.0):
        super().__init__()

        self.hidden_dim = hidden_dim
        if feedforward_dim is None:
            feedforward_dim = hidden_dim * 4
        self.feedforward_dim = feedforward_dim

        self.norm = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.fc1 = nn.Linear(hidden_dim, feedforward_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(feedforward_dim, hidden_dim)

        self.drop = nn.Dropout(drop_prob, inplace=True)
        self.add_weight = add_weight

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Feed-forward sub-block in Transformer

        :param hidden:      (batch_size, num_patches, hidden_dim)
        :return:            (batch_size, num_patches, hidden_dim)
        """
        identity = hidden

        hidden = self.norm(hidden)
        hidden = self.fc1(hidden)
        hidden = self.act(hidden)
        hidden = self.drop(hidden)
        hidden = self.fc2(hidden)
        hidden = self.drop(hidden)

        output = identity + (hidden * self.add_weight)

        return output


class PostNormFeedForwardBlock(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 feedforward_dim: Optional[int] = None,
                 drop_prob: float = 0.1,
                 act_layer=nn.GELU,
                 add_weight: float = 1.0):
        super().__init__()

        self.hidden_dim = hidden_dim
        if feedforward_dim is None:
            feedforward_dim = hidden_dim * 4
        self.feedforward_dim = feedforward_dim

        self.fc1 = nn.Linear(hidden_dim, feedforward_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(feedforward_dim, hidden_dim)
        self.drop = nn.Dropout(drop_prob, inplace=True)

        self.norm = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.add_weight = add_weight

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Feed-forward sub-block in Transformer

        :param hidden:      (batch_size, num_patches, hidden_dim)
        :return:            (batch_size, num_patches, hidden_dim)
        """
        identity = hidden

        hidden = self.fc1(hidden)
        hidden = self.act(hidden)
        hidden = self.drop(hidden)
        hidden = self.fc2(hidden)
        hidden = self.drop(hidden)

        output = identity + (hidden * self.add_weight)
        output = self.norm(output)

        return output
