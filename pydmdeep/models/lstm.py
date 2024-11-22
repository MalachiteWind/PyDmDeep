from typing import Literal

import numpy as np
import torch.nn as nn
import torch.optim.optimizer
from torch.utils.data import DataLoader

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
    epoch_test_dataset: Float2D,
    minimum_loss_decrease: float = 1e-5,
    patience: int = 10,
) -> Float1D:
    # total_train_iterations = len(dataloader) * epochs
    # loop = tqdm(total=total_train_iterations, position=0)

    d_len = epoch_test_dataset.shape[1]
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

            # loop.set_description(f"Epoch: {epoch}, train_loss: {loss.item()}")
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        epoch_loss /= len(dataloader)

        model.eval()

        with torch.no_grad():
            Vhat, Vh = reconstruct_V(
                time_delay=epoch_test_dataset,
                model=model,
                d_len=d_len,
                lags=lags,
                device=DEVICE
            )

            reconstruct_err = torch.linalg.norm(Vhat - Vh, ord="fro")
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
    # loop.close()

    return np.array(epoch_losses), np.array(reconstruct_losses)


def reconstruct_V(
    time_delay: Float2D, 
    model: LSTMModel, 
    d_len: int, 
    lags: int, 
    device: str
) -> tuple[Float2D, Float2D]:
    num_predicted_steps = d_len - lags # check if should be n_len
    _, _, Vh = np.linalg.svd(time_delay)
    Vh = torch.Tensor(Vh).to(device)
    Vhat = Vh[:lags, :]
    Vhat = torch.Tensor(Vhat)
    Vhat = Vhat.to(device)

    for i in range(num_predicted_steps):
        Vhat_sub = Vhat[i : (lags + i), :]
        Vhat_pred = model(Vhat_sub.unsqueeze(0))
        # Vhat_pred.detach_()
        Vhat = torch.vstack((Vhat, Vhat_pred))
    return Vhat, Vh
