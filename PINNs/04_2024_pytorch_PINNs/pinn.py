import functools
import matplotlib.pyplot as plt
import numpy as np
import wandb
import torch
import torch.nn as nn

import torch.utils.data as thdat
import seaborn as sns
from utils import wandb_setup
from tqdm import tqdm
from utils import check_memory_usage
sns.set_theme()
torch.manual_seed(42)

DEVICE = torch.device('mps')

def np_to_th(x):
    n_samples = len(x)
    return torch.from_numpy(x).to(torch.float).to(DEVICE).reshape(n_samples, -1)


class Net(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_units=100,
        epochs=1000,
        loss=nn.MSELoss(),
        loss2=None,
        loss2_weight=0.1,
        args=None,
    ) -> None:
        super().__init__()

        self.epochs = epochs
        self.loss = loss
        self.loss2 = loss2
        self.loss2_weight = loss2_weight
        self.loss2_args = args
        self.n_units = n_units

        self.layers = nn.Sequential(
            nn.Linear(input_dim, self.n_units),
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.ReLU(),
        )
        self.out = nn.Linear(self.n_units, output_dim)

    def forward(self, x):
        h = self.layers(x)
        out = self.out(h)

        return out
    def save(self, path, checkpoint, losses):

        torch.save(checkpoint, path)
        print(f"Saved model to {path}")

    def load(self, path, optimiser):
        print('Loading model from', path)
        checkpoint = torch.load(path, map_location=torch.device('cpu') if torch.device('cpu') is True else torch.device('cpu'))
        print('Checkpoint keys', checkpoint.keys())
        self.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['train_loss']
        losses = checkpoint['losses']

        return optimiser, epoch, loss, losses

    def fit(self, X, y, project, entity, name, optimiser):

        # Set up wandb logging, which is nice to compare runs
        metrics = wandb_setup(project=project, entity=entity,
                              name=name)

        Xt = np_to_th(X)
        yt = np_to_th(y)


        self.train()
        losses = []
        for ep in tqdm(range(0, self.epochs)):
            optimiser.zero_grad()
            outputs = self.forward(Xt)
            loss = self.loss(yt, outputs)
            if self.loss2:
                loss += self.loss2_weight + self.loss2_weight * self.loss2(self, self.loss2_args)
            loss.backward()
            optimiser.step()
            losses.append(loss.item())

            checkpoint = {
                'epoch': ep,
                'model_state_dict': self.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'train_loss': loss,
                'losses': losses
            }

            # Log with wandb
            wandb.log({"epoch": ep,
                       "train_loss": loss})
        wandb.finish()
            # if ep % int(self.epochs / 10) == 0:
            #     # print(f"Epoch {ep}/{self.epochs}, loss: {losses[-1]:.2f}")
            #     check_memory_usage(f"Epoch {ep}/{self.epochs}, loss: {losses[-1]:.2f}")
        return losses, checkpoint

    def predict(self, X):
        self.eval()
        out = self.forward(np_to_th(X))
        return out.detach().cpu().numpy()


class NetDiscovery(Net):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_units=100,
        epochs=1000,
        loss=nn.MSELoss(),
        lr=0.001,
        loss2=None,
        loss2_weight=0.1,
        args=None,
    ) -> None:
        super().__init__(
            input_dim, output_dim, n_units, epochs, loss, lr, loss2, loss2_weight, args
        )

        self.r = nn.Parameter(data=torch.tensor([0.]))

def physics_loss(model: torch.nn.Module, args):
    R, Tenv, Temps = args
    ts = torch.linspace(0, 1000, steps=1000,).view(-1,1).requires_grad_(True).to(DEVICE)
    temps = model(ts)
    dT = grad(temps, ts)[0]
    pde = R*(Tenv - temps) - dT

    return torch.mean(pde**2)

def physics_loss_pendulum(model: torch.nn.Module, args):
    ts = torch.linspace(0, 10, steps=1000,).view(-1,1).requires_grad_(True).to(DEVICE)
    thetas = model(ts)
    dTheta_dt = grad(thetas, ts)[0]
    d2Theta_dt2 = grad(dTheta_dt, ts)[0]
    pde = d2Theta_dt2 + omega**2 * torch.sin(thetas)
    return torch.mean(pde**2)

def physics_loss_discovery(model: torch.nn.Module, args):
    Tenv, Temps = args
    ts = torch.linspace(0, 1000, steps=1000,).view(-1,1).requires_grad_(True).to(DEVICE)
    temps = model(ts)
    dT = grad(temps, ts)[0]
    pde = model.r * (Tenv - temps) - dT

    return torch.mean(pde**2)
def grad(outputs, inputs):
    """Computes the partial derivative of
    an output with respect to an input.
    Args:
        outputs: (N, 1) tensor
        inputs: (N, D) tensor
    """
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
    )


def cooling_law(time, Tenv, T0, R):
    T = Tenv + (T0 - Tenv) * np.exp(-R * time)
    return T

def l2_reg(model: torch.nn.Module, arg=None):
    return torch.sum(sum([p.pow(2.) for p in model.parameters()]))


import functools
import matplotlib.pyplot as plt
import numpy as np
import wandb
import torch
import torch.nn as nn

import torch.utils.data as thdat
import seaborn as sns
from utils import wandb_setup
from tqdm import tqdm
from utils import check_memory_usage
sns.set_theme()
torch.manual_seed(42)

DEVICE = torch.device('mps')

def np_to_th(x):
    n_samples = len(x)
    return torch.from_numpy(x).to(torch.float).to(DEVICE).reshape(n_samples, -1)


class Net(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_units=100,
        epochs=1000,
        loss=nn.MSELoss(),
        loss2=None,
        loss2_weight=0.1,
        args=None,
    ) -> None:
        super().__init__()

        self.epochs = epochs
        self.loss = loss
        self.loss2 = loss2
        self.loss2_weight = loss2_weight
        self.loss2_args = args
        self.n_units = n_units

        self.layers = nn.Sequential(
            nn.Linear(input_dim, self.n_units),
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.ReLU(),
        )
        self.out = nn.Linear(self.n_units, output_dim)

    def forward(self, x):
        h = self.layers(x)
        out = self.out(h)

        return out
    def save(self, path, checkpoint, losses):

        torch.save(checkpoint, path)
        print(f"Saved model to {path}")

    def load(self, path, optimiser):
        print('Loading model from', path)
        checkpoint = torch.load(path, map_location=torch.device('cpu') if torch.device('cpu') is True else torch.device('cpu'))
        print('Checkpoint keys', checkpoint.keys())
        self.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['train_loss']
        losses = checkpoint['losses']

        return optimiser, epoch, loss, losses

    def fit(self, X, y, project, entity, name, optimiser):

        # Set up wandb logging, which is nice to compare runs
        metrics = wandb_setup(project=project, entity=entity,
                              name=name)

        Xt = np_to_th(X)
        yt = np_to_th(y)


        self.train()
        losses = []
        for ep in tqdm(range(0, self.epochs)):
            optimiser.zero_grad()
            outputs = self.forward(Xt)
            loss = self.loss(yt, outputs)
            if self.loss2:
                loss += self.loss2_weight + self.loss2_weight * self.loss2(self, self.loss2_args)
            loss.backward()
            optimiser.step()
            losses.append(loss.item())

            checkpoint = {
                'epoch': ep,
                'model_state_dict': self.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'train_loss': loss,
                'losses': losses
            }

            # Log with wandb
            wandb.log({"epoch": ep,
                       "train_loss": loss})
        wandb.finish()
            # if ep % int(self.epochs / 10) == 0:
            #     # print(f"Epoch {ep}/{self.epochs}, loss: {losses[-1]:.2f}")
            #     check_memory_usage(f"Epoch {ep}/{self.epochs}, loss: {losses[-1]:.2f}")
        return losses, checkpoint

    def predict(self, X):
        self.eval()
        out = self.forward(np_to_th(X))
        return out.detach().cpu().numpy()


class NetDiscovery(Net):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_units=100,
        epochs=1000,
        loss=nn.MSELoss(),
        lr=0.001,
        loss2=None,
        loss2_weight=0.1,
        args=None,
    ) -> None:
        super().__init__(
            input_dim, output_dim, n_units, epochs, loss, lr, loss2, loss2_weight, args
        )

        self.r = nn.Parameter(data=torch.tensor([0.]))

def physics_loss(model: torch.nn.Module, args):
    R, Tenv, Temps = args
    ts = torch.linspace(0, 1000, steps=1000,).view(-1,1).requires_grad_(True).to(DEVICE)
    temps = model(ts)
    dT = grad(temps, ts)[0]
    pde = R*(Tenv - temps) - dT

    return torch.mean(pde**2)

def physics_loss_pendulum(model: torch.nn.Module, args):
    ts = torch.linspace(0, 10, steps=1000,).view(-1,1).requires_grad_(True).to(DEVICE)
    thetas = model(ts)
    dTheta_dt = grad(thetas, ts)[0]
    d2Theta_dt2 = grad(dTheta_dt, ts)[0]
    pde = d2Theta_dt2 + omega**2 * torch.sin(thetas)
    return torch.mean(pde**2)

def physics_loss_discovery(model: torch.nn.Module, args):
    Tenv, Temps = args
    ts = torch.linspace(0, 1000, steps=1000,).view(-1,1).requires_grad_(True).to(DEVICE)
    temps = model(ts)
    dT = grad(temps, ts)[0]
    pde = model.r * (Tenv - temps) - dT

    return torch.mean(pde**2)
def grad(outputs, inputs):
    """Computes the partial derivative of
    an output with respect to an input.
    Args:
        outputs: (N, 1) tensor
        inputs: (N, D) tensor
    """
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
    )


def cooling_law(time, Tenv, T0, R):
    T = Tenv + (T0 - Tenv) * np.exp(-R * time)
    return T

def l2_reg(model: torch.nn.Module, arg=None):
    return torch.sum(sum([p.pow(2.) for p in model.parameters()]))


