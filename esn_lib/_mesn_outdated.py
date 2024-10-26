import HubESN.esn_utils.utils as utils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator

class MESN(BaseEstimator):
    def __init__(self, **kwargs):
        """
        Base Modular Echo State Network
        params:
            ## basic parameters
            lr: learning rate
            p1: average sparsity for intra-area connections
            p2: sparsity for overall connections
            n_mod: number of modules
            rho_p1: radius of submodule connectivities
                [min, max, n_mod] = [p1*(1-rho_p1), p1*(1+rho_p1), n_mod]
                equivalently np.linspace(p1-rho_p1, p1+rho_p1, n_mod)
            r_sig: ratio of signal neurons
            n_size: number of neurons
            spec_rad: spectral radius of recurrent weights
            in_features: dimension of input
            in_scale: scale of input weights
            input_neurons: where to input signals, "base", "rand", "hub"
            output_neurons: where to output signals, "all", "out"
            nonhub_type: type of non-hub neurons, "out", "mix", "in"
            use_skip: use skip connections or not
            
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
        self.p1 = kwargs.get("p1", 0.1)
        self.p2 = kwargs.get("p2", 0.1)
        self.n_mod = kwargs.get("n_mod", 1)
        self.rho_p1 = kwargs.get("rho_p1", 0)
        self.r_sig = kwargs.get("r_sig", 0.3)
        self.n_size = kwargs.get("n_size", 600)
        self.spec_rad = kwargs.get("spec_rad", 0.9)
        self.in_features = kwargs.get("in_features", 1)
        self.in_scale = kwargs.get("in_scale", 1)
        self.activation = kwargs.get("activation", "tanh")
        self.time_scale = np.ones(self.n_size) * self.lr
        self.act = utils.get_activation(self.activation)
        self.input_neurons = kwargs.get("input_neurons", "base")
        self.output_neurons = kwargs.get("output_neurons", "all")
        self.nonhub_type = kwargs.get("nonhub_type", "mix")
        self.use_skip = kwargs.get("use_skip", False)
        self.verbose = kwargs.get("verbose", False)

        # check parameters
        assert self.n_size%self.n_mod == 0, "n_size should be divisible by n_mod"
        self.mod_size = int(self.n_size / self.n_mod)
        if self.n_mod == 1: self.p1 = self.p2

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
        else:
            raise ValueError("Invalid distribution: {}".format(dist))

        # apply r_sig
        # W_mask = np.random.uniform(0, 1, size=self.n_size) > self.r_sig
        # W_ir[W_mask, :] = 0

        if self.nonhub_type == "out":
            # outdegree
            n_deg = np.count_nonzero(self.W_rc, axis=0)
        elif self.nonhub_type == "mix":
            # indegree + outdegree
            n_deg = np.count_nonzero(self.W_rc, axis=0) + np.count_nonzero(self.W_rc, axis=1)
        elif self.nonhub_type == "in":
            # indegree
            n_deg = np.count_nonzero(self.W_rc, axis=1)

        if self.input_neurons == "base":
            idx = np.argsort(n_deg)[::-1] # descending order
        elif self.input_neurons == "rand":
            idx = np.random.permutation(self.n_size)
        elif self.input_neurons == "hub":
            idx = np.argsort(n_deg) # ascending order

        if self.output_neurons == "all":
            self.out_idx = np.arange(self.n_size)
        elif self.output_neurons == "out":
            self.out_idx = idx[:int((1-self.r_sig)*self.n_size)]

        W_ir[idx[:int((1-self.r_sig)*self.n_size)], :] = 0 # set the back (1-r_sig) pecent of neurons to 0
        self.in_idx = idx[int((1-self.r_sig)*self.n_size):]
        
        self.W_ir = W_ir
        

    def _generate_wrc(self, dist, w_min, w_max):
        """
        Generate recurrent weights
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
        else:
            raise ValueError("Invalid distribution: {}".format(dist))

        # generate connectivity matrix and set weights
        self._generate_wrc_connectivity()
        self.W_rc *= w
        if self.spec_rad != -1: self.W_rc = self.W_rc * self.spec_rad / np.max(np.abs(np.linalg.eigvals(self.W_rc)))
        # self.W_rc[np.arange(self.n_size), np.arange(self.n_size)] = 0


    def _generate_wrc_connectivity(self):
        """
        Generate connectivity matrix for recurrent weights
        """
        np.random.seed(0)  # 固定随机数种子为 0
        # get number of edges
        n_edges = int(self.n_size**2 * self.p2)
        p1 = np.linspace(self.p1*(1-self.rho_p1), self.p1*(1+self.rho_p1), self.n_mod)       
        n_p1 = np.round(p1 * self.mod_size**2).astype(int) # number of edges for each module
        n_p2 = (n_edges - np.sum(n_p1))
        n_p2 *= (1 + self.mod_size**2*self.n_mod/self.n_size**2)
        p2 = n_p2 / (self.n_size**2 - self.n_size) # probability of inter-area connections

        # generate intra-area connections
        W_rc = np.random.uniform(0, 1, size=(self.n_size, self.n_size))
        W_rc = np.where(W_rc < p2, 1.0, 0.0)
        for i in range(self.n_mod):
            _w = np.random.uniform(0, 1, size=(self.mod_size, self.mod_size))
            _w = np.where(_w < p1[i], 1.0, 0.0)
            W_rc[i*self.mod_size:(i+1)*self.mod_size, i*self.mod_size:(i+1)*self.mod_size] = _w
        
        self.W_rc = W_rc
    

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
        states_stack = np.hstack((states, inputs)) if self.use_skip else states

        self.W_ro = np.dot(np.linalg.pinv(states_stack[transient:, self.out_idx]), labels[transient:, :]).T
        # remember the last state for later:
        self.laststate = states[-1, :]
        self.lastinput = inputs[-1, :]
        self.lastoutput = labels[-1, :]
        self.out_features = labels.shape[1]

        # apply learned weights to the collected states:
        pred_train = np.dot(states_stack[:, self.out_idx], self.W_ro.T)
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
            t1 = np.concatenate([states[n + 1, self.out_idx], inp]) if self.use_skip else states[n + 1, self.out_idx]
            outputs[n + 1, :] = t1 @ self.W_ro.T

            if use_last_state:
                inp = outputs[n + 1, :]
            else:
                inp = inputs[n+1, :] if n < n_samples - 1 else None

        return outputs[1:], states[1:]


    def _get_area_edges(self):
        n_intra = np.zeros((self.n_mod, ))
        for a in range(self.n_mod):
            n = self.mod_size
            w_area = self.W_rc[a*n:(a+1)*n, a*n:(a+1)*n]
            n_intra[a] = np.count_nonzero(w_area)
        
        n_inter = np.count_nonzero(self.W_rc) - np.sum(n_intra)
        return n_intra, n_inter


    def plot_weight(self):
        """
        Plot W_rc weight matrix
        """
        rad = np.max(np.abs(self.W_rc))
        plt.figure(figsize=(10, 10))
        plt.imshow(self.W_rc, cmap="bwr", vmin=-rad, vmax=rad)
        plt.colorbar()
        plt.show()


    def get_save_dict(self):
        specs = {}
        not_to_print = ["time_scale", "W_rc", "W_ir", "act", "in_idx", "W_ro", "laststate", "lastinput", "lastoutput", "out_features", "out_idx"]
        for key, val in self.__dict__.items():
            if key not in not_to_print: specs[key] = val
        
        n_intra, n_inter = self._get_area_edges()
        specs["intra_mod_spar"] = n_intra/self.mod_size**2
        specs["inter_mod_spar"] = n_inter/(self.n_size**2 - self.n_mod*self.mod_size**2)
        specs["total_spar"] = (n_intra.sum() + n_inter)/(self.n_size**2)
        return specs
    

    def print_layer(self):
        specs = self.get_save_dict()
        utils.print_params("MESN", specs)


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