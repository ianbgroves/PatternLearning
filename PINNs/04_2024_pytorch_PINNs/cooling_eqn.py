import wandb

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

t, T, Tenv, t0, R, timecourse, eq, temps = generate_and_plot_training_data()

# Train an unregularised normal feedforward neural network with inputs t, T

net = Net(1,1, loss2=None, epochs=20000, lr=1e-5).to(DEVICE)
print('Training with no regularisation:')
losses = net.fit(t, T)

plt.plot(losses)
plt.yscale('log')
plt.show()

# Add l2 regularisation, which reduces overfitting
netreg = Net(1,1, loss2=l2_reg, epochs=20000, lr=1e-4, loss2_weight=1).to(DEVICE)
print('Training with l2 regularisation:')
losses = netreg.fit(t, T)

plt.plot(losses)
plt.yscale('log')

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
plt.show()