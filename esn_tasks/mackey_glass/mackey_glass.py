import os
import numpy as np
from scipy import signal

class MackeyGlass():
    def __init__(self, *args, **kwargs):
        """
        This class generate the Mackey-Glass time series prediction task.
        """
        super().__init__()
        self.data_path = kwargs.get('data_path', os.path.join(os.path.dirname(__file__), 'mackey_glass.npy'))
        self.n_train = kwargs.get('n_train', 1000)
        self.n_test = kwargs.get('n_test', 1000)
        self.delay = kwargs.get('delay', 1)

        self._init_task()

    def _init_task(self):
        """
        Load the data from the file.
        """
        data = np.load(self.data_path).reshape(-1, 1)
        self.X_train = data[:self.n_train]
        self.y_train = data[self.delay:self.n_train+self.delay]
        self.X_test = data[self.n_train:self.n_train+self.n_test]
        self.y_test = data[self.n_train+self.delay:self.n_train+self.n_test+self.delay]

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
            if np.abs(pred[i]) > 1:
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
