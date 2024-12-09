import optuna
import optunahub

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_iris

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit


data = load_iris()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x = data["data"]
# Binarize target.
y = np.where(data["target"] >= 1, 1, 0)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=45)

class Net(nn.Module):
# class Net(jit.ScriptModule):
    def __init__(self, L2: int, L3: int):
        """
        Initialization of the neural network architecture
        with a specified amount of layers and neurons.
        Tipp: Sometimes it helps to use batch normalization layers 
        to standardize inputs for the next layer and thus 
        making training more stable and faster.
        """
        super().__init__()
        self.fc1 = nn.Linear(4, L2)
        self.fc2 = nn.Linear(L2, L3)
        self.fc3 = nn.Linear(L3, 1)
    # @jit.script_method
    def forward(self, x):
        """
        Feeding input x through the layers and activation functions (relu, sigmoid).
        Returning output y after propagating through the whole architecture.
        """
        y = F.sigmoid(self.fc1(x))
        y = F.sigmoid(self.fc2(y))
        y = F.sigmoid(self.fc3(y))
        return y
    
loss = nn.BCELoss()
# @jit.script_method
def objective (trial: optuna.Trial) -> float:
    model = Net(trial.suggest_int("L2",1,8),trial.suggest_int("L3",1,8))
    optimizer = torch.optim.Adam(model.parameters(), lr=trial.suggest_float("lr", 1.e-3, 1, log="True"))

    # L1_norm = torch.tensor(0., requires_grad=True)
    # L1_lambda = trial.suggest_float("L1_lambda",1.e-5,1,log="True")

    for step in range(100):
        optimizer.zero_grad()
        error = loss(model(torch.from_numpy(x_train).float()), 
                        torch.from_numpy(y_train).unsqueeze(1).float())
        # L1_norm = torch.tensor(0., requires_grad=True)
        # L1_norm = sum(torch.linalg.norm(param, 1) for name, param in model.named_parameters() if "weight" in name)
        # error += L1_lambda*L1_norm
        error.backward()
        optimizer.step()
        trial.report(error.item(), step)
        if trial.should_prune():
            raise optuna.TrialPruned() 
    return(error.item())

study = optuna.create_study(
        study_name="distributed-example", 
        storage="sqlite:///example.db",
        # sampler=optunahub.load_module("samplers/auto_sampler").AutoSampler(),
        sampler=optuna.RandomSampler(),
        pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=30, interval_steps=10),
        load_if_exists=True
    )
# study = optuna.create_study(sampler=optunahub.load_module("samplers/auto_sampler").AutoSampler())
study.optimize(objective, n_trials=1000, n_jobs=-1)

print(study.best_trial.value, study.best_trial.params)