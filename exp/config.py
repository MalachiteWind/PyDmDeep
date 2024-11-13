import torch.nn as nn

tensor_data_lookup = {
    "seed": {"default": 1234},
    "lags": {"default": 30},
    "train_len": {"default": 0.8},
}


lstm_train_lookup = {"loss": {"mse": nn.MSELoss()}}
