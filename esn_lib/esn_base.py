import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator

import sys
sys.path.append("../esn_utils")
# import HubESN.esn_utils.modularity as mod
import esn_utils.modularity as mod
import networkx as nx
import bct
np.random.seed(0)  # 设置随机种子为0
class ESNBase(BaseEstimator):
    def __init__(self, **kwargs):
        """
        Base Modular Echo State Network
        params:
            ## basic parameters
            lr: learning rate
            p2: sparsity for overall connections
            r_sig: ratio of signal neurons
            n_size: number of neurons
            spec_rad: spectral radius of recurrent weights
            in_features: dimension of input
            in_scale: scale of input weights
            input_neurons: where to input signals, "peri", "rand", "hub"
            output_neurons: where to output signals, "all", "out"
            hub_type: definition of hub neurons, "out", "mix", "in"
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
        self.lr = kwargs.get("lr", 0.5)
        self.p2 = kwargs.get("p2", 0.1)
        self.r_sig = kwargs.get("r_sig", 0.3)
        self.n_size = kwargs.get("n_size", 500)
        self.spec_rad = kwargs.get("spec_rad", 0.9)
        self.in_features = kwargs.get("in_features", 1)
        self.in_scale = kwargs.get("in_scale", 1)
        self.activation = kwargs.get("activation", "tanh")
        self.act = self.get_activation(self.activation)
        self.input_neurons = kwargs.get("input_neurons", "rand")
        self.output_neurons = kwargs.get("output_neurons", "all")
        self.hub_type = kwargs.get("hub_type", "mix")
        self.use_skip = kwargs.get("use_skip", False)
        self.verbose = kwargs.get("verbose", False)

        self.not_to_print = [
            "W_rc", "W_rc_mask", "W_ir", "W_ir_mask", "W_ro", "act", 
            "in_idx", "out_idx","degree_idx", "out_features", "laststate", "lastinput", "lastoutput",
            "not_to_print", "x", "y", "z"]

        self.rc_dist = kwargs.get("rc_dist", "normal")
        self.rc_min = kwargs.get("rc_min", -1)
        self.rc_max = kwargs.get("rc_max", 1)
        self.in_dist = kwargs.get("in_dist", "uniform")
        self.in_min = kwargs.get("in_min", -1)
        self.in_max = kwargs.get("in_max", 1)


    def _generate_wir(self):
        """
        Generate input weights
        The input shape: (*, in_features)
        Apply a linear transformation to the incoming data: y = xA^T
        """
        np.random.seed(0)  # 固定随机数种子为 0
        dist, w_min, w_max = self.in_dist, self.in_min, self.in_max
        # get random input weights
        if (dist == "uniform"):
            W_ir = np.random.uniform(w_min, w_max, size=(self.n_size, self.in_features))
        elif (dist == "normal"):
            _mu = (w_min + w_max) / 2
            _sigma = (w_max - w_min) / 6
            W_ir = np.random.normal(_mu, _sigma, size=(self.n_size, self.in_features))
        else:
            raise ValueError("Invalid distribution: {}".format(dist))
        self.W_ir = W_ir


    def _generate_wir_mask(self):
        # set the back (1-r_sig) pecent of neurons to 0
        if self.hub_type == "out":
            # outdegree
            n_deg = np.count_nonzero(self.W_rc, axis=0)
        elif self.hub_type == "mix":
            # indegree + outdegree
            n_deg = np.count_nonzero(self.W_rc, axis=0) + np.count_nonzero(self.W_rc, axis=1)
        elif self.hub_type == "in":
            # indegree
            n_deg = np.count_nonzero(self.W_rc, axis=1)

        # sort neurons by their degree
        if self.input_neurons == "peri":
            idx = np.argsort(n_deg)[::-1] # descending order
        elif self.input_neurons == "rand":
            idx = np.random.permutation(self.n_size)
        elif self.input_neurons == "hub":
            idx = np.argsort(n_deg) # ascending order

        self.degree_idx =idx
        # set the back (1-r_sig) pecent of neurons to 0 or not
        if self.output_neurons == "all":
            self.out_idx = np.arange(self.n_size)
        elif self.output_neurons == "out":
            self.out_idx = idx[:int((1-self.r_sig)*self.n_size)]

        # generate mask
        W_ir_mask = np.ones((self.n_size, self.in_features))

        W_ir_mask[idx[:int((1-self.r_sig)*self.n_size)], :] = 0 # set the back (1-r_sig) pecent of neurons to 0

        self.in_idx = idx[int((1-self.r_sig)*self.n_size):]
        self.W_ir_mask = W_ir_mask
        self.W_ir = self.W_ir * self.W_ir_mask


        

    def _generate_wrc(self):
        """
        Generate recurrent weights
        """
        np.random.seed(0)  # 固定随机数种子为 0
        dist, w_min, w_max = self.rc_dist, self.rc_min, self.rc_max
        # get random recurrent weights
        # 均匀分布
        if (dist == "uniform"):
            W_rc = np.random.uniform(w_min, w_max, (self.n_size, self.n_size))
        # 正太分布
        elif (dist == "normal"):
            _mu = (w_min + w_max) / 2
            _sigma = (w_max - w_min) / 6
            W_rc = np.random.normal(_mu, _sigma, (self.n_size, self.n_size))
        else:
            raise ValueError("Invalid distribution: {}".format(dist))
        self.W_rc = W_rc


    def _generate_wrc_mask(self):
        """
        Generate recurrent weights
        """
        W_rc_mask = np.zeros((self.n_size, self.n_size))

        # randomly select neurons to be connected
        n_conn = int(self.n_size**2 * self.p2)
        conn_from = np.random.choice(self.n_size, n_conn, replace=True)
        conn_to = np.random.choice(self.n_size, n_conn, replace=True)

        W_rc_mask[conn_from, conn_to] = 1
        self.W_rc_mask = W_rc_mask
        self.W_rc = self.W_rc * self.W_rc_mask


    def _apply_spec_rad(self):
        """
        Set the spectral radius of the recurrent weight matrix
        """
        # set the spectral radius
        if self.spec_rad != -1: self.W_rc = self.W_rc * self.spec_rad / np.max(np.abs(np.linalg.eigvals(self.W_rc)))
        # self.W_rc[np.arange(self.n_size), np.arange(self.n_size)] = 0
        

    def get_activation(self, act):
        if act == "tanh": return np.tanh
        elif act == "relu": return lambda x: np.maximum(0, x)
        elif act == "sigmoid": return lambda x: 1 / (1 + np.exp(-x))

        
    def _update(self, state, inputs):
        """
        Update the state of the network
        params:
            state: previous state of the network, shape: (n_size,)
            inputs: input data, shape: (in_features,)
        """
        preactivation = state @ self.W_rc.T + self.in_scale * inputs @ self.W_ir.T
        state = (1 - self.lr) * state + self.lr * self.act(preactivation)
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
            states[n, :] = self._update(states[n - 1].reshape(1, -1), inputs[n,:].reshape(1, -1))
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
        for n in tqdm(range(inputs.shape[0]),
                      desc="Training…",
                      ascii=False, ncols=75,
                      disable=not self.verbose):
            states[n, :] = self._update(states[n - 1].reshape(1, -1), inputs[n,:].reshape(1, -1))
        
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
            states[n + 1, :] = self._update(states[n, :].reshape(1, -1), inp.reshape(1, -1))
            t1 = np.concatenate([states[n + 1, self.out_idx], inp]) if self.use_skip else states[n + 1, self.out_idx]
            outputs[n + 1, :] = t1 @ self.W_ro.T

            if use_last_state:
                inp = outputs[n + 1, :]
            else:
                inp = inputs[n+1, :] if n < n_samples - 1 else None

        return outputs[1:], states[1:]

    def predict_custom(self, inputs,W_ro, use_last_state=False):
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
            states[n + 1, :] = self._update(states[n, :].reshape(1, -1), inp.reshape(1, -1))
            t1 = np.concatenate([states[n + 1, self.out_idx], inp]) if self.use_skip else states[n + 1, self.out_idx]
            outputs[n + 1, :] = t1 @ W_ro.T

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


    def get_degrees(self, norm=False, type='all'):
        if type == "in": deg = np.count_nonzero(self.W_rc, axis=1)
        elif type == "out": deg = np.count_nonzero(self.W_rc, axis=0)
        else: deg = np.count_nonzero(self.W_rc, axis=1) + np.count_nonzero(self.W_rc, axis=0)

        if norm:
            return (deg - np.min(deg)) / (np.max(deg) - np.min(deg))
        else:
            return deg

    def global_efficiency(self):
        '''
        Return：
        特征路径长度、全局效率、偏心率、半径、直径：
        + 特征路径长度是网络中的平均最短路径长度
        + 全局效率是网络中的平均逆最短路径长度
        + 节点偏心率是节点与任何其他节点之间的最大最短路径长度
        + 半径是最小偏心率，直径是最大偏心率

        '''
        weight_matrix = self.W_rc
        binary_weight_matrix = weight_matrix.copy().astype(int)
        thresh = np.quantile(weight_matrix, q=0.9)
        matrix_mask = weight_matrix > thresh
        binary_weight_matrix[matrix_mask] = 1
        binary_weight_matrix[~matrix_mask] = 0

        d, e = bct.distance_wei(binary_weight_matrix)

        l, eff, ecc, radius, diameter = bct.charpath(d)
        return l, eff, ecc, radius, diameter
    def get_heterogeneity(self):
        """ 
        Calculate the coefficient of variation based on the absolute value of the weights
        params:
            W: weight matrix, assume it is updated using y = xA^T
            where x is the state vector with shape (1, n_size)
        """
        deg = np.count_nonzero(self.W_rc, axis=1) + np.count_nonzero(self.W_rc, axis=0)
        deg = deg[deg > 0]
        return np.std(deg) / np.mean(deg)


    def _get_params(self):
        params = {}
        for key, val in self.__dict__.items():
            if key not in self.not_to_print: params[key] = val
        
        params["unconnected_neurons"] = self.get_unconnected_neurons()
        params["in_spar"] = np.round(np.count_nonzero(self.W_ir) / self.n_size / self.in_features, 4)
        params["total_spar"] = np.round(np.count_nonzero(self.W_rc) / self.n_size**2, 4)
        params["heterogeneity"] = np.round(self.get_heterogeneity(), 4)
        return params


    def get_modularity(self):
        """
        Calculate the modularity of the network
        """
        num_uc = self.get_unconnected_neurons()
        # assert num_uc == 0, "There are {} unconnected neurons".format(num_uc)
        if num_uc != 0:
            # print("There are {} unconnected neurons".format(num_uc))
            return 0
        else:
            return mod.compute_modularity(self.W_rc)


    def get_unconnected_neurons(self):
        connected = np.count_nonzero(self.W_rc, axis=1) + np.count_nonzero(self.W_rc, axis=0)
        connected = np.count_nonzero(connected)
        return self.n_size - connected


    def plot_dist(self):
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


    def plot_eigen(self):
        deg = np.count_nonzero(self.W_rc, axis=0) + np.count_nonzero(self.W_rc, axis=1)
        deg = deg / np.max(deg)

        w, v = np.linalg.eig(self.W_rc)
        x = [ele.real for ele in w]
        # extract imaginary part
        y = [ele.imag for ele in w]

        # plot the complex numbers
        plt.figure(figsize=(6, 4.8))
        plt.tight_layout()
        plt.scatter(x, y, c=deg, cmap='bwr')
        plt.colorbar()
        plt.ylabel('Imaginary')
        plt.xlabel('Real')
        plt.title('Eigenvalues of W_rc')
        plt.show()


    def get_eigen_centralities(self):
        """
        Calculate the eigen centrality of the network
        """
        eigenvalues, eigenvectors = np.linalg.eig(self.W_rc)
        largest_eigenvalue_index = np.argmax(eigenvalues)
        # Get the eigenvector corresponding to the largest eigenvalue
        eigenvector_centrality = eigenvectors[:, largest_eigenvalue_index]

        # Normalize the eigenvector centrality
        eigenvector_centrality_normalized = eigenvector_centrality / np.sum(eigenvector_centrality)

        return eigenvector_centrality_normalized


    def create_directed_graph_from_weight_matrix(self, W):
        G = nx.DiGraph()
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                if W[i, j] > 0:
                    G.add_edge(i, j, weight=W[i, j])
        return G
    

    def get_clustering_coefficient(self):
        """
        Calculate the clustering coefficient of the network
        """
        G = self.create_directed_graph_from_weight_matrix(self.W_rc)
        clustering_coeffs = nx.clustering(G, weight='weight')
        return np.mean(list(clustering_coeffs.values()))
    
    def get_spec_rad(self):
        """
        Calculate the spectral radius of the network
        """
        return np.max(np.abs(np.linalg.eigvals(self.W_rc)))
    
    def get_avg_path_length(self):
        """
        Calculate the average path length of the network
        """
        G = self.create_directed_graph_from_weight_matrix(self.W_rc)
        G_un = G.to_undirected()
        path_lengths = nx.all_pairs_dijkstra_path_length(G_un, weight='weight')
        total_path_length = 0
        num_paths = 0
        for source, paths in path_lengths:
            num_paths += len(paths) - 1
            total_path_length += sum(paths.values()) - paths[source]

        return total_path_length / num_paths