import HubESN.esn_utils.utils as utils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator

class SFHESN(BaseEstimator):
    def __init__(self, **kwargs):
        """
        Scale-Free Hub ESN
        params:
            ## basic parameters
            lr: learning rate
            p2: sparsity for overall connections
            r_sig: ratio of input to recurrent connections
            n_size: number of neurons
            spec_rad: spectral radius of recurrent weights
            in_features: dimension of input
            in_scale: scale of input weights
            sf_dist: scale free distriubtion shape, positive exponent sf_dist - 1
            sf_scale: scale largest degree of the scale free distribution (10**sf_scale)
            n_levels: number of levels of the scale free distribution
            input_neurons: where to input signals, 0: base neurons, 1: random, 2: hub neurons
            
            ## weight initialization parameters
            rc_dist: distribution of recurrent weights, "uniform" or "normal"
            rc_min: minimum value of recurrent weights
            rc_max: maximum value of recurrent weights
            in_dist: distribution of input weights, "uniform" or "normal"
            in_min: minimum value of input weights
            in_max: maximum value of input weights
            activation: activation function, "tanh" or "relu"
            verbose: print the progress of training
        """
        
        # parameters
        self.lr = kwargs.get("lr", 0.1)
        self.p2 = kwargs.get("p2", 0.1)
        self.r_sig = kwargs.get("r_sig", 0.3)
        self.n_size = kwargs.get("n_size", 600)
        self.spec_rad = kwargs.get("spec_rad", 0.9)
        self.in_features = kwargs.get("in_features", 1)
        self.in_scale = kwargs.get("in_scale", 1)
        self.sf_dist = kwargs.get("sf_dist", 20)
        self.sf_scale = kwargs.get("sf_scale", 3)
        self.n_levels = kwargs.get("n_levels", 20)
        self.activation = kwargs.get("activation", "tanh")
        self.time_scale = np.ones(self.n_size) * self.lr
        self.act = utils.get_activation(self.activation)
        self.input_neurons = kwargs.get("input_neurons", 2)
        self.verbose = kwargs.get("verbose", False)

        # initialize EI_Balanced_ESN
        self._generate_wrc(kwargs.get("rc_dist", "uniform"), 
                           kwargs.get("rc_min", -1), 
                           kwargs.get("rc_max", 1))
        self._generate_wir(kwargs.get("in_dist", "uniform"), 
                           kwargs.get("in_min", -1), 
                           kwargs.get("in_max", 1))


    def _generate_wir(self, dist, w_min, w_max):
        """
        Generate input weights
        The input shape: (*, in_features)
        Apply a linear transformation to the incoming data: y = xA^T
        params:
            dist: distribution of input weights, "uniform" or "normal"
            w_min: minimum value of input weights
            w_max: maximum value of input weights
        """
        np.random.seed(0)  # 固定随机数种子为 0
        if (dist == "uniform"):
            W_ir = np.random.uniform(w_min, w_max, size=(self.n_size, self.in_features))
        elif (dist == "normal"):
            _mu = (w_min + w_max) / 2
            _sigma = (w_max - w_min) / 6
            W_ir = np.random.normal(_mu, _sigma, size=(self.n_size, self.in_features))

        n_deg = np.count_nonzero(self.W_rc, axis=0)
        if self.input_neurons == 0:
            idx = np.argsort(n_deg)[::-1] # descending order
        elif self.input_neurons == 1:
            idx = np.random.permutation(self.n_size)
        elif self.input_neurons == 2:
            idx = np.argsort(n_deg) # ascending order

        W_ir[idx[:int((1-self.r_sig)*self.n_size)], :] = 0 # set the back (1-r_sig) pecent of neurons to 0
        self.in_idx = idx[int((1-self.r_sig)*self.n_size):]

        self.W_ir = W_ir


    def _generate_wrc(self, dist, w_min, w_max):
        """
        Generate recurrent weights
        The input shape: (*, n_size)
        Apply a linear transformation to the incoming data: y = xA^T + b
        params:
            dist: distribution of recurrent weights, "uniform" or "normal"
            w_min: minimum value of recurrent weights
            w_max: maximum value of recurrent weights
        """
        np.random.seed(0)  # 固定随机数种子为 0
        # get random recurrent weights
        if (dist == "uniform"):
            w = np.random.uniform(w_min, w_max, (self.n_size, self.n_size))
        elif (dist == "normal"):
            _mu = (w_min + w_max) / 2
            _sigma = (w_max - w_min) / 6
            w = np.random.normal(_mu, _sigma, (self.n_size, self.n_size))

        # generate connectivity matrix and set weights
        self._generate_wrc_connectivity()
        self.W_rc *= w
        if self.spec_rad != -1: self.W_rc = self.W_rc * self.spec_rad / np.max(np.abs(np.linalg.eigvals(self.W_rc)))
        self.W_rc[np.arange(self.n_size), np.arange(self.n_size)] = 0


    def _generate_wrc_connectivity(self):
        """
        Generate recurrent weights connectivity
        """
        s = np.random.power(self.sf_dist, self.n_size)
        counts, bins = np.histogram(s, bins=self.n_levels)
        counts = counts[::-1]
        
        n_total = 0 # total number of edges before scale
        n_edges = int(self.p2*self.n_size*self.n_size) # number of edges after scale
        for i, c in enumerate(counts):
            n_total += c * 10**(i*self.sf_scale/self.n_levels)
        x = n_edges//n_total

        edge_steps = np.round(np.array([10**(i*self.sf_scale/self.n_levels)*x for i in range(self.n_levels)])/2) # half for in-degree and half for out-degree
        edge_steps = edge_steps.astype(int)
        # plt.plot(edge_steps, counts)
        # plt.show()
        
        # generate connectivity matrix
        self.W_rc = np.zeros((self.n_size, self.n_size))
        cur_neuron = 0
        for i, c in enumerate(counts):
            for _ in range(c):
                # randomly select n indexes
                n = edge_steps[i]
                in_idx = np.random.choice(self.n_size, size=n, replace=False)
                out_idx = np.random.choice(self.n_size, size=n, replace=False)
                self.W_rc[cur_neuron, out_idx] = 1
                self.W_rc[in_idx, cur_neuron] = 1
                cur_neuron += 1

        for i in range(self.n_size):
            self.W_rc[i, i] = 0

        # # shuffle the matrix
        # idx = np.arange(self.n_size)
        # np.random.shuffle(idx)
        # self.W_rc = self.W_rc[idx, :]
        # self.W_rc = self.W_rc[:, idx]


    def _update(self, state, inputs):
        """
        Update the state of the network
        params:
            state: previous state of the network, shape: (n_size,)
            inputs: input data, shape: (in_features,)
        """
        # print(state.dtype, inputs.dtype, self.W_rc.dtype, self.W_ir.dtype, self.time_scale.dtype)
        preactivation = state @ self.W_rc.T + self.in_scale * inputs @ self.W_ir.T
        state = (1 - self.time_scale) * state + self.time_scale * self.act(preactivation)
        return state


    def run(self, inputs):
        """
        Run the network
        params:
            inputs: input data, shape: (n_samples, in_features)
        """
        # states = np.random.normal(0, 0.1**2, size=(inputs.shape[0], self.n_size))
        states = np.zeros((inputs.shape[0], self.n_size))
        for n in range(inputs.shape[0]):
            states[n, :] = self._update(states[n - 1].reshape(1, -1), inputs[n,:].reshape(1, -1)) + states[n, :].reshape(1, -1)
        return states


    def fit(self, inputs, labels):
        """
        Fit the EI_Balanced_ESN
        params:
            inputs: input data, shape: (n_samples, in_features)
            labels: label data, shape: (n_samples, out_features)
        """
        # states = np.random.normal(0, 0.1**2, size=(inputs.shape[0], self.n_size))
        states = np.zeros((inputs.shape[0], self.n_size))
        for n in tqdm(range(1, inputs.shape[0]),
                      desc="Training…",
                      ascii=False, ncols=75,
                      disable=not self.verbose):
            states[n, :] = self._update(states[n - 1].reshape(1, -1), inputs[n,:].reshape(1, -1)) + states[n, :].reshape(1, -1)
        
        transient = min(int(inputs.shape[0] / 10), 100)
        states_stack = np.hstack((states, inputs))

        self.W_ro = np.dot(np.linalg.pinv(states_stack[transient:, :]), labels[transient:, :]).T
        # remember the last state for later:
        self.laststate = states[-1, :]
        self.lastinput = inputs[-1, :]
        self.lastoutput = labels[-1, :]
        self.out_features = labels.shape[1]

        # apply learned weights to the collected states:
        pred_train = np.dot(states_stack, self.W_ro.T)
        return pred_train



    def predict(self, inputs, use_last_state=False):
        """
        Predict the output
        params:
            inputs: input data, shape: (n_samples, in_features)
        """
        n_samples = inputs.shape[0]
        states = np.vstack([self.laststate, np.zeros((n_samples, self.n_size))])
        outputs = np.vstack([self.lastoutput, np.zeros((n_samples, self.out_features))])

        inp = inputs[0, :]
        for n in tqdm(range(n_samples),
                      desc="Predicting…",
                      ascii=False, ncols=75,
                      disable=not self.verbose):
            states[n + 1, :] = self._update(states[n, :].reshape(1, -1), inp)
            t1 = np.concatenate([states[n + 1, :], inp])
            outputs[n + 1, :] = t1 @ self.W_ro.T

            if use_last_state:
                inp = outputs[n + 1, :]
            else:
                inp = inputs[n+1, :] if n < n_samples - 1 else None

        return outputs[1:], states[1:]


    def plot_weight(self):
        """
        Plot W_rc weight matrix
        """
        rad = np.max(np.abs(self.W_rc))
        plt.figure(figsize=(10, 10))
        plt.imshow(self.W_rc, cmap="bwr", vmin=-rad, vmax=rad)
        plt.colorbar()
        plt.show()


    def print_layer(self):
        specs = {}
        not_to_print = ["time_scale", "W_rc", "W_ir", "act", "in_idx", "W_ro", "laststate", "lastinput", "lastoutput", "out_features"]
        for key, val in self.__dict__.items():
            if key not in not_to_print: specs[key] = val
        
        n_inter = np.count_nonzero(self.W_rc)
        n_zero_deg = np.count_nonzero(self.W_rc, axis=0) + np.count_nonzero(self.W_rc, axis=1)
        n_zero_deg = np.count_nonzero(n_zero_deg == 0)
        specs["total_spar"] = n_inter/(self.n_size**2)
        specs["r_sig"] = np.count_nonzero(self.W_ir, axis=0)[0]/self.n_size
        specs["zero_deg"] = n_zero_deg

        utils.print_params("SFHESN", specs)


    def plot_degree_dist(self):
        """
        Plot the indegree distribution
        """
        indeg = np.count_nonzero(self.W_rc, axis=1)
        count, _ = np.histogram(indeg, bins=100)
        plt.figure(figsize=(6, 3))
        plt.bar(np.arange(100), count)
        plt.xlabel("Indegree")
        plt.ylabel("Count")
        plt.yticks(np.linspace(0, count.max(), 5))
        plt.show()


    def _get_area_edges(self):
        n_intra = np.zeros((self.n_mod, ))
        for a in range(self.n_mod):
            n = self.mod_size
            w_area = self.W_rc[a*n:(a+1)*n, a*n:(a+1)*n]
            n_intra[a] = np.count_nonzero(w_area)
        
        n_inter = np.count_nonzero(self.W_rc) - np.sum(n_intra)
        return n_intra, n_inter