import numpy as np
class RecallTask():
    def __init__(self, *args, **kwargs):
        """
        Recall task. Below description assumes bit_len = 5, n_bit = 4, T0 = 20.
        -   The total length of a single run is (10+T0). There are (n_bit+2) binary input 
            channels and (n_bit+2) binary output channels. The first n_bit channels of each of the
            inputs or outputs carries the memory pattern. The (n_bit+1)th channel in the input feeds
            the distractor input. The (n_bit + 2)th input channel carries the cue.
        -   In the output, the third output channel should signal the “waiting for recall cue” 
            condition, and the fourth is unused and should always be zero (this channel could just 
            as well be dropped but it is included in the original task specs, so I keep it).
        -   Specificially, an input sequence is constructed as follows. First, all (n_bit + 2)x(bit_len x 2 + T0)
            inputs are initialized to zero. Then, for the first 5 timesteps, one of the first two
            input channels is randomly set to 1. Note that there are altogether $4^5 = 1024$
            possible input patterns. Then, for timesteps 6 through T0 + 4, the third input channel 
            is set to 1 (distractor signal). At time T0 + 5, the fourth input channel is set to 1 
            (cue). For the remaining times [T0 + 6, T0 + 10], the input is again set to 1 on 
            the 3rd channel. Thus, at every timestep exactly one of the four inputs is 1.
        -   The target output is always zero on all channels, except for times 1 to T0 + 5,
            where it is 1 on the 3rd channel, and for times T0 + 6 to the end where the input
            memory pattern is repeated in the first two channels.

        params:
            n_trials: the number of trials, -1 if iterates over all possible sequences
            test_ratio: the ratio of testing data
            T0: the length of the distractor signal
            n_bit: the bit of the sequence (e.g. 4 for 4-bit memory task)
            bit_len: the length of the sequence (e.g. 5 for 5 x n-bit memory task)
            shuffle: whether to shuffle the data
        """
        self.n_trials = kwargs.get('n_trials', -1)
        self.test_ratio = kwargs.get('test_ratio', 0.2)
        self.T0 = kwargs.get('T0', 20)
        self.n_bit = kwargs.get('n_bit', 4)
        self.bit_len = kwargs.get('bit_len', 5)
        self.shuffle = kwargs.get('shuffle', True)

        self._init_task()


    def _init_task(self):
        """
        Generate the recall task training and testing data.
        return:
            X_train: the training input
            y_train: the training target output
            X_test: the testing input
            y_test: the testing target output
        """
        if self.n_trials == -1:
            seq_list = self._generate_recall_task_all()
        else:
            seq_list = self._generate_recall_task_random()

        if self.shuffle:
            np.random.shuffle(seq_list)

        n_train = int(len(seq_list) * (1 - self.test_ratio))
        train_seq, test_seq = seq_list[:n_train], seq_list[n_train:]

        self.X_train, self.y_train = self._seq_to_data(train_seq)
        self.X_test, self.y_test = self._seq_to_data(test_seq)


    def _seq_to_data(self, seq):
        """
        Convert a sequence to the input and target output.
        params:
            seq: the sequence to be recalled, shape (n_trials, bit_len)
        return:
            X: the input
            y: the target output
        """
        X, y = [], []
        for s in seq:
            _X, _y = self._generate_recall_trial(s)
            X.append(_X)
            y.append(_y)
        if len(X) != 0:
            X, y = np.vstack(X), np.vstack(y)
        return np.array(X), np.array(y)


    def get_data(self):
        """
        Get the training and testing data.
        return:
            X_train: the training input
            y_train: the training target output
            X_test: the testing input
            y_test: the testing target output
        """

        return self.X_train, self.y_train, self.X_test, self.y_test


    def _data_to_seq(self, y):
        """
        Convert the output to the sequence.
        params:
            y: the output
        return:
            seq: the sequence
        """
        trial_len = self.bit_len*2 + self.T0
        assert y.shape[0] % trial_len == 0, "The prediction length is not correct"

        seq = []
        for i in range(int(y.shape[0]/trial_len)):
            _y = y[(i+1)*trial_len-self.bit_len:(i+1)*trial_len, :self.n_bit]
            _y = np.argmax(_y, axis=1)
            seq.append(_y)
        return np.array(seq)


    def _generate_recall_trial(self, seq):
        """
        Generate a single trial for recall task.
        params:
            seq: the sequence to be recalled
        return
            X: the input
            y: the target output
        """
        T0 = self.T0
        n_bit = self.n_bit
        bit_len = self.bit_len

        X = np.zeros((T0+bit_len*2, n_bit+2))
        X[np.arange(bit_len), seq] = 1
        X[np.arange(bit_len, T0+bit_len*2), n_bit] = 1
        X[T0+bit_len-1, n_bit+1] = 1
        X[T0+bit_len-1, n_bit] = 0

        y = np.zeros((T0+bit_len*2, n_bit+2))
        y[np.arange(T0+bit_len), n_bit] = 1
        y[np.arange(T0+bit_len, T0+bit_len*2), seq] = 1

        return X, y
    
    
    def _generate_recall_task_all(self):
        """
        Iterate over all possible sequences.
        return:
            seq_list = the list of sequences
        """
        seq_list = []
        for i in range(self.n_bit**self.bit_len):
            _seq = np.array([int(x) for x in np.base_repr(i, base=self.n_bit).zfill(self.bit_len)])
            seq_list.append(_seq)
        return np.array(seq_list)


    def _generate_recall_task_random(self):
        """
        Generate the n recall seq randomly.
        return:
            seq_list = the list of sequences
        """
        seq_list = []
        for i in range(self.n_trials):
            _seq = np.random.randint(self.n_bit, size=self.bit_len)
            seq_list.append(_seq)
        return np.array(seq_list)
    

    def eval(self, y_pred, y_true):
        """
        Evaluate the prediction.
        params:
            y_pred: predicted value
            y_true: true value
        """

        seq_pred = self._data_to_seq(y_pred)
        seq_true = self._data_to_seq(y_true)
        count = 0
        mse = 0
        for i in range(len(seq_pred)):
            if np.array_equal(seq_pred[i], seq_true[i]):
                count += 1
            mse += np.mean((seq_pred[i] - seq_true[i])**2)

        return count/seq_pred.shape[0], mse/seq_pred.shape[0]