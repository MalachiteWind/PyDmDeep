import torch.nn as nn
import torch.optim as optim

tensor_data_lookup = {}


lstm_train_lookup = {
    "num_layers": {"small": 10},
    "num_epochs": {"1k": 1000, "2k": 2000},
    "opt": {
        "SGD": (optim.SGD, {"lr": 0.01}),
        "SGD_05": (optim.SGD, {"lr": 0.05}),
        }, # (opt, opt keywords)
    "loss": {"MSE": nn.MSELoss},
    "dataloader_kws": {"small_batch": {"batch_size": 50, "shuffle": True}},
    "model_trainer_kws": {
        "low_min_decrease": {"minimum_loss_decrease": 1e-8, "patience":10}
    }
}

dmd_lookup = {}
