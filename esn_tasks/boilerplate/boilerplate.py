import numpy as np
class ESNTasks():
    def __init__(self, *args, **kwargs):
        """
        This class is a base class for all ESN tasks.
        """
        pass

    def _init_task(self):
        """
        Initialize the task.
        """
        pass

    def get_data(self):
        """
        Return the training and testing data.
        """
        pass

    def eval(self, y_pred, y_true):
        """
        Evaluate the prediction.
        params:
            y_pred: predicted value
            y_true: true value
        """
        pass