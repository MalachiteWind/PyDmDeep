from typing import Literal

import numpy as np
import torch.nn as nn
import torch.optim.optimizer
from torch.utils.data import DataLoader

from ..types import Float1D


DEVICE = "cuda" if torch.cuda.is_available() else "CPU"


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def model_trainer(
    model: LSTMModel,
    epochs: int,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_criterion: nn.Module,
    device: Literal["cuda", "CPU"],
    minimum_loss_decrease: float = 1e-5,
    patience: int = 10,
) -> Float1D:
    # total_train_iterations = len(dataloader) * epochs
    # loop = tqdm(total=total_train_iterations, position=0)

    best_loss = np.inf
    patience_counter = 0
    epoch_losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            # x = x.cuda(non_blocking=True).float()
            # y = y.cuda(non_blocking=True).long()

            optimizer.zero_grad()
            target_pred = model(data)
            loss = loss_criterion(target, target_pred)

            # loop.set_description(f"Epoch: {epoch}, train_loss: {loss.item()}")
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        epoch_loss /= len(dataloader)

        if best_loss - epoch_loss >= minimum_loss_decrease:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > patience:
            print(
                f"Loss decrease threshold reached.\
                      Epoch: {epoch +1}. Loss: {epoch_loss}."
            )
            epoch_losses.append(epoch_loss)
            break
        epoch_losses.append(epoch_loss)
    # loop.close()

    return np.array(epoch_losses)
