import numpy as np
from torchvision.datasets import mnist
from sklearn.preprocessing import OneHotEncoder

class MNIST784():
    """ 
    *** Part of the code is from https://blog.csdn.net/itnerd/article/details/109230929 
    MNIST task, each image is reshaped into a 1x784 vector.
    """
    def __init__(self, *args, **kwargs):
        """
        This class is a base class for all ESN tasks.
        """
        self.n_train = kwargs.get('n_train', 1000)
        self.n_test = kwargs.get('n_test', 1000)
        self.w = 28 # width of the image
        self.l = 10 # number of labels

        self._init_task()

    def _init_task(self):
        """
        Initialize the task.
        """
        train_set = mnist.MNIST('./data', train=True, download=True)
        test_set = mnist.MNIST('./data', train=False, download=True)

        train_data = train_set.data.numpy()/255
        train_labels = train_set.targets.numpy().reshape(-1,1)
        enc = OneHotEncoder()
        enc.fit(train_labels)
        train_labels = enc.transform(train_labels).toarray()

        test_data = test_set.data.numpy()/255
        test_labels = test_set.targets.numpy().reshape(-1,1)
        test_labels = enc.transform(test_labels).toarray()

        w = self.w
        l = self.l

        self.X_train = train_data[:self.n_train].reshape(self.n_train*w*w, -1)
        self.y_train = np.array([np.array([train_labels[i]]*w*w) for i in range(self.n_train)]).reshape(self.n_train*w*w, -1)
        self.X_test = test_data[:self.n_test].reshape(self.n_test*w*w, -1)
        self.y_test = np.array([np.array([test_labels[i]]*w*w) for i in range(self.n_test)]).reshape(self.n_test*w*w, -1)
        

    def get_data(self):
        """
        Return the training and testing data.
        """
        return self.X_train, self.y_train, self.X_test, self.y_test
        

    def to_label(self, y):
        """ Convert duplicated on-hot labels to regular labels. """
        y_ = y.reshape(-1, self.w*self.w, self.l).mean(axis=1)
        # y_ = y_[:, -24:, :].mean(axis=1)
        return np.argmax(y_, axis=1)

    def eval(self, y_pred, y_true):
        """
        Evaluate the prediction.
        params:
            y_pred: predicted value
            y_true: true value
        """
        y_pred = self.to_label(y_pred)
        y_true = self.to_label(y_true)
        return np.mean(y_pred == y_true)