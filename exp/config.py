import torch.nn as nn
import torch.optim as optim

tensor_data_lookup = {
    "seed": {"default": 1234},
    "lags": {"default": 30},
    "train_len": {"default": 0.8},
}


lstm_train_lookup = {
    "num_layers": {"small": 10},
    "num_epochs": {"1k": 1000},
    "opt": {"SGD": (optim.SGD, {"lr": 0.01})},
    "loss": {"MSE": nn.MSELoss},
    "dataloader_kws": {"small_batch": {"batch_size": 50, "shuffle": True}},
}
