from scipy.integrate import odeint
import matplotlib.pyplot as plt
from pinn import *
import os
import torch
import torch.optim as optim
# Set up wandb logging, which is nice to compare runs
metrics = wandb_setup(project='cooling_pinns', entity='shefpinns',
                      name='nonlinear_pendulum')

os.makedirs('models/nonlinear_pendulum/', exist_ok=True)
def second_order_ode(y, t, omega):
    theta, omega_theta = y
    dydt = [omega_theta, -omega**2 * np.sin(theta)]  # Second-order ODE
    return dydt


device = torch.device('mps')
np.random.seed(10)

# Initial conditions
theta_0 = 0.5  # initial displacement
omega_theta_0 = 0.0  # initial velocity
y0 = [theta_0, omega_theta_0]

# Time points to solve the ODE for
times = np.linspace(0, 10, 1000)

# Solve the ODE
omega = 2.0  # frequency parameter
sol = odeint(second_order_ode, y0, times, args=(omega,))

# Make training data
t = np.linspace(0, 10, 50)
theta = np.interp(t, times, sol[:, 0]) +  0.025 * np.random.randn(50)

# Plot the results
plt.figure()
plt.plot(times, sol[:, 0])
plt.plot(t, theta, 'o')
plt.show()

learning_rate = 1e-5


net = Net(1,1, loss2=None, epochs=100000).to(DEVICE)
optimiser = optim.Adam(net.parameters(), lr=learning_rate)
load = False
if load:
    optimiser, epoch, loss, losses = net.load(path='models/nonlinear_pendulum/nonlinear_pendulum_vanilla.pth',
             optimiser=optimiser)
else:
    losses, checkpoint = net.fit(t, theta, project='cooling_pinns',
                                 entity='shefpinns', name='nonlinear_pendulum_vanilla',
                                 optimiser=optimiser)

    net.save('models/nonlinear_pendulum/nonlinear_pendulum_vanilla.pth', checkpoint, losses)

plt.figure()
plt.plot(losses)
plt.yscale('log')
plt.show()

netreg = Net(1,1, loss2=l2_reg, epochs=100000, loss2_weight=1).to(DEVICE)
optimiser = optim.Adam(netreg.parameters(), lr=learning_rate)

if load:
    optimiser, epoch, loss, losses = net.load(path='models/nonlinear_pendulum/nonlinear_pendulum_l2reg.pth',
                                              optimiser=optimiser)
else:

    losses, checkpoint = netreg.fit(t, theta, project='cooling_pinns', entity='shefpinns',
                                    name='nonlinear_pendulum_l2_reg', optimiser=optimiser)

plt.figure()
plt.plot(losses)
plt.yscale('log')


def physics_loss(model: torch.nn.Module, args):
    ts = torch.linspace(0, 10, steps=1000,).view(-1,1).requires_grad_(True).to(DEVICE)
    thetas = model(ts)
    dTheta_dt = grad(thetas, ts)[0]
    d2Theta_dt2 = grad(dTheta_dt, ts)[0]
    pde = d2Theta_dt2 + omega**2 * torch.sin(thetas)
    return torch.mean(pde**2)

pinn = Net(1, 1, loss2=physics_loss, epochs=100000, loss2_weight=1, args=()).to(DEVICE)
optimiser = optim.Adam(pinn.parameters(), lr=learning_rate)
losses, checkpoint = pinn.fit(t, theta, project='cooling_pinns', entity='shefpinns', name='nonlinear_pendulum_pinn_reg', optimiser=optimiser)
plt.figure()
plt.plot(losses)
plt.yscale('log')

predsreg = netreg.predict(times)
preds = net.predict(times)
predspinn = pinn.predict(times)

plt.figure()
plt.plot(times, sol[:, 0], alpha=0.8, label='Equation')
plt.plot(t, theta, 'o', label='Training data')
plt.plot(times, preds, alpha=0.8, label='Network')
plt.plot(times, predsreg, alpha=0.8, label='L2 Network')
plt.plot(times, predspinn, alpha=0.8, label='PINN')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

wandb.finish()