import os
import numpy as np
from scipy import signal


np.random.seed(0)  # 设置随机种子为0
class MackeyGlass():
    def __init__(self, *args, **kwargs):
        """
        This class generate the Mackey-Glass time series prediction task.
        """
        super().__init__()
        self.n_train = kwargs.get('n_train', 1000)
        self.n_test = kwargs.get('n_test', 1000)
        self.tau = kwargs.get('tau', 17)
        self.delay = kwargs.get('delay', 1)
        self.include_tau = kwargs.get('include_tau', False)

        self._init_task()

    def _init_task(self):
        """
        Load the data from the file.
        """
        if self.include_tau:
            data = self.generate_mackey(self.n_train+self.n_test+self.tau+self.delay).reshape(-1, 1)
            # set X to be f(n) and f(n-tau), y to be f(n+tau+1)
            self.X_train = np.concatenate((data[:self.n_train], data[self.tau:self.n_train+self.tau]), axis=1)
            self.y_train = data[self.tau+self.delay:self.n_train+self.tau+self.delay]
            self.X_test = np.concatenate((data[self.n_train:self.n_train+self.n_test], data[self.n_train+self.tau:self.n_train+self.n_test+self.tau]), axis=1)
            self.y_test = data[self.n_train+self.tau+self.delay:self.n_train+self.n_test+self.tau+self.delay]
        else:
            data = self.generate_mackey(self.n_train+self.n_test+self.delay).reshape(-1, 1)
            self.X_train = data[:self.n_train]
            self.y_train = data[self.delay:self.n_train+self.delay]
            self.X_test = data[self.n_train:self.n_train+self.n_test]
            self.y_test = data[self.n_train+self.delay:self.n_train+self.n_test+self.delay]

    def generate_mackey(self, N):
        gamma   = 0.1
        beta   = 0.2
        tau = self.tau

        y = [0.9697, 0.9699, 0.9794, 1.0003, 1.0319, 1.0703, 1.1076, 1.1352, 1.1485,
            1.1482, 1.1383, 1.1234, 1.1072, 1.0928, 1.0820, 1.0756, 1.0739, 1.0759]

        for n in range(17,N+99):
            y.append(y[n] - gamma*y[n] + beta*y[n-tau]/(1+y[n-tau]**10))

        mg = np.array(y[100:])

        # normalize mg to -1 to 1
        mg = (mg - np.min(mg))/(np.max(mg) - np.min(mg)) * 2 - 1

        return mg
    
    def get_data(self):
        """
        Return the training and testing data.
        """
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def eval(self, y_pred, y_true):
        """
        Evaluate the prediction.
        params:
            y_pred: predicted value
            y_true: true value
        """
        if len(y_true) < len(y_pred):
            print(f'Warning: y_true is shorter than y_pred, y_pred is truncated from {len(y_pred)} to {len(y_true)}')
            y_pred = y_pred[:len(y_true)]
        return np.mean(np.abs(y_pred - y_true))
    
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
    
    def time_avg_rmse(self, y_pred, y_true):
        """
        Evaluate the prediction using time averaged RMSE.
        params:
            y_pred: predicted value
            y_true: true value
        """
        if len(y_true) < len(y_pred):
            print(f'Warning: y_true is shorter than y_pred, y_pred is truncated from {len(y_pred)} to {len(y_true)}')
            y_pred = y_pred[:len(y_true)]
        return np.sqrt(np.mean(np.square(y_pred - y_true)))/len(y_true)
    
    def correlation(self, y_pred, y_true):
        """
        Evaluate the prediction using correlation.
        params:
            y_pred: predicted value
            y_true: true value
        """
        if len(y_true) < len(y_pred):
            print(f'Warning: y_true is shorter than y_pred, y_pred is truncated from {len(y_pred)} to {len(y_true)}')
            y_pred = y_pred[:len(y_true)]
        
            corr = signal.correlate(y_pred, y_true, mode='full')
            return np.max(corr)
    
    def get_pred_length(self, pred):
        for i in range(len(pred)):
            if np.abs(pred[i]) > 1.1:
                return i
        return len(pred)
    
    def effective_pred_length(self, y_pred, y_true):
        """
        Evaluate the prediction using effective prediction length.
        Return the position where difference between predicted value and true value is larger than 0.5
        params:
            y_pred: predicted value
            y_true: true value
        """
        if len(y_true) < len(y_pred):
            print(f'Warning: y_true is shorter than y_pred, y_pred is truncated from {len(y_pred)} to {len(y_true)}')
            y_pred = y_pred[:len(y_true)]
        
        # return the position where difference between predicted value and true value is larger than 0.5
        for i in range(len(y_pred)):
            if np.abs(y_pred[i] - y_true[i]) > 0.5:
                return i
        return len(y_pred)
