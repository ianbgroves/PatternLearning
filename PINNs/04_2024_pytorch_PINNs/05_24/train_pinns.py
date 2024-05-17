import functools
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR
import seaborn as sns
from scipy.integrate import odeint

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def np_to_th(x):
    n_samples = len(x)
    return torch.from_numpy(x).to(torch.float).to(DEVICE).reshape(n_samples, -1)


def create_network(input_dim, output_dim, n_units=100):
    layers = nn.Sequential(
        nn.Linear(input_dim, n_units),
        nn.ReLU(),
        nn.Linear(n_units, n_units),
        nn.ReLU(),
        nn.Linear(n_units, n_units),
        nn.ReLU(),
        nn.Linear(n_units, n_units),
        nn.ReLU(),
    )
    out = nn.Linear(n_units, output_dim)
    return layers, out


def train_network(layers, out, X, y, epochs=1000, loss_fn=nn.MSELoss(), lr=1e-3, regularize=False, pde_loss=None,
                  pde_weight=0.1):
    Xt, yt = np_to_th(X), np_to_th(y)
    optimizer = optim.Adam(list(layers.parameters()) + list(out.parameters()), lr=lr)

    scheduler = StepLR(optimizer, step_size=2000, gamma=0.5)  # Change learning rate by a factor of +- gamma every 2000 epochs

    losses = []
    for ep in range(epochs):
        optimizer.zero_grad()
        h = layers(Xt)
        outputs = out(h)
        loss = loss_fn(yt, outputs)

        if regularize:
            l2_reg = sum(torch.sum(p.pow(2.0)) for p in layers.parameters()) + sum(
                torch.sum(p.pow(2.0)) for p in out.parameters())
            loss += 0.001 * l2_reg

        if pde_loss:
            loss += pde_weight * pde_loss(layers, out)

        loss.backward()
        optimizer.step()

        # Update the learning rate
        scheduler.step()

        losses.append(loss.item())

        if ep % (epochs // 10) == 0:
            print(f"Epoch {ep}/{epochs}, loss: {losses[-1]:.2f}, lr: {scheduler.get_last_lr()[0]:.6f}")

    return losses


def predict(layers, out, X):
    with torch.no_grad():
        Xt = np_to_th(X)
        h = layers(Xt)
        outputs = out(h)
    return outputs.cpu().numpy()


def grad(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)


def cooling_law(time, Tenv, T0, R):
    T = Tenv + (T0 - Tenv) * np.exp(-R * time)
    return T


def temp_pde_loss(layers, out, R, Tenv):
    ts = torch.linspace(0, 1000, steps=1000, ).view(-1, 1).requires_grad_(True).to(DEVICE)
    temps = out(layers(ts))
    dT = grad(temps, ts)[0]
    pde = R * (Tenv - temps) - dT
    return torch.mean(pde ** 2)


def second_order_ode(y, t, omega):
    theta, omega_theta = y
    dydt = [omega_theta, -omega ** 2 * np.sin(theta)]
    return dydt


def pendulum_pde_loss(layers, out, omega):
    ts = torch.linspace(0, 10, steps=1000, ).view(-1, 1).requires_grad_(True).to(DEVICE)
    thetas = out(layers(ts))
    dTheta_dt = grad(thetas, ts)[0]
    d2Theta_dt2 = grad(dTheta_dt, ts)[0]
    pde = d2Theta_dt2 + omega ** 2 * torch.sin(thetas)
    return torch.mean(pde ** 2)

def temperature_prediction(n_epochs, learning_rate):
    # Temperature prediction example
    Tenv, T0, R = 25, 100, 0.005
    times = np.linspace(0, 1000, 1000)
    eq = functools.partial(cooling_law, Tenv=Tenv, T0=T0, R=R)
    temps = eq(times)

    t = np.linspace(0, 300, 10)
    T = eq(t) + 2 * np.random.randn(10)

    layers, out = create_network(1, 1)
    losses = train_network(layers, out, t, T, epochs=n_epochs, lr=learning_rate)
    preds = predict(layers, out, times)
    torch.save(layers.state_dict(), 'temp_layers.pth')
    torch.save(out.state_dict(), 'temp_out.pth')

    layers_reg, out_reg = create_network(1, 1)
    losses_reg = train_network(layers_reg, out_reg, t, T, epochs=n_epochs, lr=learning_rate, regularize=True)
    preds_reg = predict(layers_reg, out_reg, times)
    torch.save(layers_reg.state_dict(), 'temp_layers_reg.pth')
    torch.save(out_reg.state_dict(), 'temp_out_reg.pth')

    layers_pinn, out_pinn = create_network(1, 1)
    losses_pinn = train_network(layers_pinn, out_pinn, t, T, epochs=n_epochs, lr=learning_rate,
                                pde_loss=lambda layers, out: temp_pde_loss(layers, out, R, Tenv),
                                pde_weight=1)
    preds_pinn = predict(layers_pinn, out_pinn, times)
    torch.save(layers_pinn.state_dict(), 'temp_layers_pinn.pth')
    torch.save(out_pinn.state_dict(), 'temp_out_pinn.pth')

    # Plotting for temperature prediction
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Normal')
    plt.plot(losses_reg, label='L2 Regularized')
    plt.plot(losses_pinn, label='PINN')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(times, temps, alpha=0.8, label='Equation')
    plt.plot(t, T, 'o', label='Training data')
    plt.plot(times, preds, alpha=0.8, label='Network')
    plt.plot(times, preds_reg, alpha=0.8, label='L2 Network')
    plt.plot(times, preds_pinn, alpha=0.8, label='PINN')
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (C)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def nonlinear_pendulum_prediction(n_epochs, vanilla_lr, l2_lr, pinn_lr, num_training_points, pde_weight):
    # Nonlinear pendulum example
    np.random.seed(10)
    theta_0, omega_theta_0 = 0.5, 0.0
    y0 = [theta_0, omega_theta_0]
    times_pendulum = np.linspace(0, 10, 1000)
    omega = 2.0
    sol = odeint(second_order_ode, y0, times_pendulum, args=(omega,))

    t_pendulum = np.linspace(0, 10, num_training_points)
    theta = np.interp(t_pendulum, times_pendulum, sol[:, 0]) + 0.025 * np.random.randn(num_training_points)

    layers_pendulum, out_pendulum = create_network(1, 1)
    print('Vanilla network')
    losses_pendulum = train_network(layers_pendulum, out_pendulum, t_pendulum, theta, epochs=n_epochs, lr=vanilla_lr)
    preds_pendulum = predict(layers_pendulum, out_pendulum, times_pendulum)
    torch.save(layers_pendulum.state_dict(), 'pendulum_layers.pth')
    torch.save(out_pendulum.state_dict(), 'pendulum_out.pth')

    layers_pendulum_reg, out_pendulum_reg = create_network(1, 1)
    print('L2 regularised network')
    losses_pendulum_reg = train_network(layers_pendulum_reg, out_pendulum_reg, t_pendulum, theta, epochs=n_epochs,
                                        lr=l2_lr,
                                        regularize=True)
    preds_pendulum_reg = predict(layers_pendulum_reg, out_pendulum_reg, times_pendulum)
    torch.save(layers_pendulum_reg.state_dict(), 'pendulum_layers_reg.pth')
    torch.save(out_pendulum_reg.state_dict(), 'pendulum_out_reg.pth')

    print('PDE regularised network')
    layers_pendulum_pinn, out_pendulum_pinn = create_network(1, 1)
    losses_pendulum_pinn = train_network(layers_pendulum_pinn, out_pendulum_pinn, t_pendulum, theta, epochs=n_epochs,
                                         lr=pinn_lr, pde_loss=lambda layers, out: pendulum_pde_loss(layers, out, omega), pde_weight=pde_weight)
    preds_pendulum_pinn = predict(layers_pendulum_pinn, out_pendulum_pinn, times_pendulum)
    torch.save(layers_pendulum_pinn.state_dict(), 'pendulum_layers_pinn.pth')
    torch.save(out_pendulum_pinn.state_dict(), 'pendulum_out_pinn.pth')

    # Plotting for nonlinear pendulum
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses_pendulum, label='Normal')
    plt.plot(losses_pendulum_reg, label='L2 Regularized')
    plt.plot(losses_pendulum_pinn, label='PINN')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(bottom=1e-7)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(times_pendulum, sol[:, 0], alpha=0.8, label='Equation')
    plt.plot(t_pendulum, theta, 'o', label='Training data')
    plt.plot(times_pendulum, preds_pendulum, alpha=0.8, label='Network')
    plt.plot(times_pendulum, preds_pendulum_reg, alpha=0.8, label='L2 Network')
    plt.plot(times_pendulum, preds_pendulum_pinn, alpha=0.8, label='PINN')
    plt.xlabel('Time')
    plt.ylabel('Angle')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sns.set_theme()
    torch.manual_seed(42)
    # temperature_prediction(n_epochs=30000, learning_rate=1e-4)
    nonlinear_pendulum_prediction(n_epochs=20000, vanilla_lr=1e-4, l2_lr=1e-4, pinn_lr=1e-5, num_training_points=100, pde_weight=1)

