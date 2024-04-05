import functools
import numpy as np
import matplotlib.pyplot as plt
from pinn import cooling_law
def generate_and_plot_training_data(plot=True):
    np.random.seed(10)

    Tenv = 25
    T0 = 100
    R = 0.005
    times = np.linspace(0, 1000, 1000)
    eq = functools.partial(cooling_law, Tenv=Tenv, T0=T0, R=R)
    temps = eq(times)

    # Make training data
    t = np.linspace(0, 300, 10)
    T = eq(t) +  2 * np.random.randn(10)

    if plot:
        plt.plot(times, temps)
        plt.plot(t, T, 'o')
        plt.legend(['Equation', 'Training data'])
        plt.ylabel('Temperature (C)')
        plt.xlabel('Time (s)')
        plt.show()
    return t, T, Tenv, T0, R, times, eq, temps