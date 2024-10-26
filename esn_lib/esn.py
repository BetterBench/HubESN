import numpy as np
# import HubESN.esn_utils.utils as utils
# from HubESN.esn_lib.esn_base import *
import esn_utils.utils as utils
from esn_lib.esn_base import *

np.random.seed(0)  # 设置随机种子为0
class ESN(ESNBase):
    def __init__(self, **kwargs):
        """
        Echo State Network
        """
        super().__init__(**kwargs)

        # initialize EI_Balanced_ESN
        self._generate_wrc()
        self._generate_wrc_mask()
        self._generate_wir()
        self._generate_wir_mask()

        self._apply_spec_rad()


    def _generate_wrc_connectivity(self):
        """
        Generate connectivity matrix
        """
        # randomly select neurons to be connected
        n_conn = int(self.n_size**2 * self.p2)
        conn_from = np.random.choice(self.n_size, n_conn, replace=True)
        conn_to = np.random.choice(self.n_size, n_conn, replace=True)

        self.W_rc = np.zeros((self.n_size, self.n_size))
        self.W_rc[conn_from, conn_to] = 1


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
    

    def get_save_dict(self):
        params = self._get_params()
        return params
    

    def print_layer(self):
        utils.print_params("ESN", self.get_save_dict())