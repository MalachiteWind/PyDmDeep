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
    lags: int,
    V_test_dataset: Float2D,
    minimum_loss_decrease: float = 1e-5,
    patience: int = 10,
) -> Float1D:
    '''
    V_test_dataset: shape = (nt,nx)
    '''
    t_len = V_test_dataset.shape[0]
    best_loss = np.inf
    patience_counter = 0
    epoch_losses = []
    reconstruct_losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            target_pred = model(data)
            loss = loss_criterion(target, target_pred)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        epoch_loss /= len(dataloader)

        model.eval()

        with torch.no_grad():
            Vhat, V = reconstruct_V(
                V_scaled=V_test_dataset,
                model=model,
                t_len=t_len,
                lags=lags,
                device=DEVICE
            )

            reconstruct_err = torch.linalg.norm(Vhat - V, ord="fro") / torch.linalg.norm(V,ord="fro")
            reconstruct_losses.append(reconstruct_err.item())
        
        model.train()

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

    return np.array(epoch_losses), np.array(reconstruct_losses)


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
