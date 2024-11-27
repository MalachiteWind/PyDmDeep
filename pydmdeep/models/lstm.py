from typing import Literal
from typing import Optional

import numpy as np
import torch.nn as nn
import torch.optim.optimizer
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from ..types import Float1D
from ..types import Float2D

DEVICE = "cuda" if torch.cuda.is_available() else "CPU"


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
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
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_criterion: nn.Module,
    device: Literal["cuda", "CPU"],
    minimum_loss_decrease: float = 1e-5,
    patience: int = 10,
) -> tuple[Float1D,Float1D]:
    '''
    train lstm model.
    '''
  
    best_loss = np.inf
    patience_counter = 0
    epoch_losses = []
    val_error = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for data, target in train_dataloader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            target_pred = model(data)
            loss = loss_criterion(target, target_pred)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        epoch_loss /= len(train_dataloader)

        model.eval()
        with torch.no_grad():
            epoch_err = 0
            for data, target in val_dataloader:
                data, target = data.to(device), target.to(device)

                target_pred = model(data)
                epoch_err += torch.linalg.norm(
                    target-target_pred,ord = "fro") / torch.linalg.norm(target)
    
            val_error.append(epoch_err.cpu() / len(val_dataloader))


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

    return np.array(epoch_losses), np.array(val_error)


def reconstruct_V(
    V_scaled: Float2D, 
    model: LSTMModel, 
    t_len: int, 
    lags: int, 
    device: str,
) -> tuple[Float2D, Float2D]:
    '''
    time_delay: shape=(nx,nt)
    '''
    num_predicted_steps = t_len - lags 
    V = torch.Tensor(V_scaled).to(device)
    Vhat = V[:lags, :]
    Vhat = torch.Tensor(Vhat)
    Vhat = Vhat.to(device)

    for i in range(num_predicted_steps):
        Vhat_sub = Vhat[i : (lags + i), :]
        Vhat_pred = model(Vhat_sub.unsqueeze(0))
        Vhat = torch.vstack((Vhat, Vhat_pred))
    return Vhat, V
