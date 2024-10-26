import os
import numpy as np
np.random.seed(0)  # 设置随机种子为0
class NARMA10():
    def __init__(self, *args, **kwargs):
        """
        This class generate the Mackey-Glass time series prediction task.
        """
        super().__init__()
        self.n_train = kwargs.get('n_train', 1000)
        self.n_test = kwargs.get('n_test', 1000)

        self._init_task()

    def _init_task(self):
        """
        Load the data from the file.
        """
        u, y = self.generate_narma(self.n_train+self.n_test)
        self.X_train = u[:self.n_train]
        self.y_train = y[:self.n_train]
        self.X_test = u[self.n_train:self.n_train+self.n_test]
        self.y_test = y[self.n_train:self.n_train+self.n_test]

    def generate_narma(self, N):
        """
        Generate a NARMA10 time series.
        """
        # Initialize the input and output series
        np.random.seed(0)  # 固定随机数种子为 0
        u = np.random.uniform(0, 0.5, N)
        y = np.zeros(N)

        # Generate the NARMA10 series
        for t in range(10, N):
            y[t] = 0.3*y[t-1] + 0.05*y[t-1]*np.sum(y[t-10:t]) + 1.5*u[t-10]*u[t-1] + 0.1

        return u.reshape(-1, 1), y.reshape(-1, 1)
    
    def get_data(self):
        """
        Return the training and testing data.
        """
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def eval(self, y_pred, y_true):
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