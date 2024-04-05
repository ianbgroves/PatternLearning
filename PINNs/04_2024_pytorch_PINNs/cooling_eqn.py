import wandb
import os
import torch
import sklearn
import numpy as np
import pickle as pkl
from pinn import *
from utils import check_memory_usage, wandb_setup
from data import generate_and_plot_training_data

# Set up wandb logging, which is nice to compare runs
# metrics = wandb_setup(project='cooling_pinns', entity='shefpinns',
#                       name='cooling_eqn')

check_memory_usage('Script start')
os.makedirs('models/cooling_eqn/', exist_ok=True)
t, T, Tenv, t0, R, timecourse, eq, temps = generate_and_plot_training_data()

# Train an unregularised normal feedforward neural network with inputs t, T

net = Net(1,1, loss2=None, epochs=30000, lr=1e-5).to(DEVICE)
print('Training with no regularisation:')
losses = net.fit(t, T, project='cooling_pinns', entity='shefpinns', name='cooling_eqn_vanilla')
net.save('models/cooling_eqn/vanilla_net.pth')

plt.plot(losses)
plt.yscale('log')
# plt.show()

# Add l2 regularisation, which reduces overfitting
netreg = Net(1,1, loss2=l2_reg, epochs=30000, lr=1e-4, loss2_weight=1).to(DEVICE)
print('Training with l2 regularisation:')
losses = netreg.fit(t, T, project='cooling_pinns', entity='shefpinns', name='cooling_eqn_vanilla_l2reg')
netreg.save('models/cooling_eqn/reg_net.pth')
plt.plot(losses)
plt.yscale('log')
# plt.show()

# predict with both networks for comparison
preds = net.predict(timecourse)
predsreg = netreg.predict(timecourse)

plt.figure()
plt.plot(timecourse, temps, alpha=0.8)
plt.plot(t, T, 'o')
plt.plot(timecourse, preds, alpha=0.8)
plt.plot(timecourse, predsreg, alpha=0.8)

plt.legend(labels=['Equation','Training data', 'Network', 'L2 Network'])
plt.ylabel('Temperature (C)')
plt.xlabel('Time (s)')
# plt.show()

pinn = Net(1,1, loss2=physics_loss, epochs=100000, loss2_weight=1, lr=1e-5, args=(R, Tenv, temps)).to(DEVICE)

print('Training with PINN regularisation')
losses = pinn.fit(t, T, project='cooling_pinns', entity='shefpinns', name='cooling_eqn_pinn')
pinn.save('models/cooling_eqn/pinn.pth')
plt.figure()
plt.plot(losses)
plt.yscale('log')
# plt.show()

preds = pinn.predict(timecourse)

plt.figure()
plt.plot(timecourse, temps, alpha=0.8)
plt.plot(t, T, 'o')
plt.plot(timecourse, preds, alpha=0.8)
plt.legend(labels=['Equation','Training data', 'PINN'])
plt.ylabel('Temperature (C)')
plt.xlabel('Time (s)')
# plt.show()

netdisc = NetDiscovery(1, 1, loss2=physics_loss_discovery, loss2_weight=1, epochs=100000, lr= 5e-6, args=(Tenv, temps)).to(DEVICE)
print('Finding parameter values')
losses = netdisc.fit(t, T, project='cooling_pinns', entity='shefpinns', name='cooling_eqn_discovery')
netdisc.save('models/cooling_eqn/netdisc.pth')
plt.figure()
plt.plot(losses)
plt.yscale('log')
# plt.show()

preds = netdisc.predict(timecourse)
print(netdisc.r)

plt.plot(timecourse, temps, alpha=0.8)
plt.plot(t, T, 'o')
plt.plot(timecourse, preds, alpha=0.8)
plt.legend(labels=['Equation','Training data', 'discovery PINN'])
plt.ylabel('Temperature (C)')
plt.xlabel('Time (s)')
# plt.show()