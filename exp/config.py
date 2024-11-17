import torch.nn as nn
import torch.optim as optim

tensor_data_lookup = {}


lstm_train_lookup = {
    "num_layers": {"small": 10},
    "num_epochs": {"1k": 1000},
    "opt": {"SGD": (optim.SGD, {"lr": 0.01})},
    "loss": {"MSE": nn.MSELoss},
    "dataloader_kws": {"small_batch": {"batch_size": 50, "shuffle": True}},
}
