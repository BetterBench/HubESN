import os
import numpy as np
from scipy.integrate import odeint

class Lorenz():
    def __init__(self, *args, **kwargs):
        """
        This class generate the Mackey-Glass time series prediction task.
        """
        super().__init__()
        self.T = kwargs.get('T', 100)
        self.ratio = kwargs.get('ratio', 0.8)

        self._init_task()

    def _init_task(self):
        """
        Load the data from the file.
        """
        u, y = self.generate_lorenz_prediction_task(self.T)
        n_train = int(self.ratio * len(u))

        self.X_train = u[:n_train]
        self.y_train = y[:n_train]
        self.X_test = u[n_train:]
        self.y_test = y[n_train:]


    def lorenz_system(self, state, t, sigma, rho, beta):
        x, y, z = state
        return sigma*(y - x), x*(rho - z) - y, x*y - beta*z
    
    def generate_lorenz_prediction_task(self, T, init_state=[1.0, 1.0, 1.0], sigma=10.0, rho=28.0, beta=8.0/3, dt=0.01):
        """
        Generate a Lorenz system prediction task.
        T: total time
        init_state: initial state of the system
        sigma, rho, beta: parameters of the Lorenz system
        dt: time step size
        """
        t = np.arange(0, T, dt)
        trajectory = odeint(self.lorenz_system, init_state, t, args=(sigma, rho, beta))

        # The task is to predict the next state given the current state
        inputs = trajectory[:-1]
        targets = trajectory[1:]

        print('inputs', inputs.shape)
        print('targets', targets.shape)
        return inputs, targets
    
    def get_data(self):
        """
        Return the training and testing data.
        """
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def rmse(self, y_pred, y_true):
        """
        Evaluate the prediction using RMSE.
        params:
            y_pred: predicted value
            y_true: true value
        """
        if len(y_true) < len(y_pred):
            print(f'Warning: y_true is shorter than y_pred, y_pred is truncated from {len(y_pred)} to {len(y_true)}')
            y_pred = y_pred[:len(y_true)]
        return np.sqrt(np.mean(np.square(y_pred - y_true)))
