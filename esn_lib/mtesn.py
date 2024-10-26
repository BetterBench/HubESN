import numpy as np
import ZhaozeWang.HubESN.esn_utils.utils as utils
from HubESN.esn_lib.esn import *


class MTESN(ESN):
    def __init__(self, **kwargs):
        """
        Bio-realistic Multi-Tasking Hub Echo State Network
        params:
        """
        kwargs["in_features"] = 1
        super().__init__(**kwargs)

        self.n_task = kwargs.get("n_task", 2)
        self._generate_mt_wir()


    def _generate_mt_wir(self):
        """
        Generate multi-tasking wir
        """
        input_idx = np.where(np.all(self.W_ir_mask, axis=1))[0]
        np.random.shuffle(input_idx)
        n_per_task = int(self.r_sig * self.n_size / self.n_task)

        W_ir_list = []
        for i in range(self.n_task):
            W_ir = self.W_ir.copy()
            W_ir[input_idx[i * n_per_task: (i + 1) * n_per_task]] = 0
            W_ir_list.append(W_ir)

        self.not_to_print.append("W_ir_list")
        self.W_ir_list = W_ir_list
        

    def _update(self, state, inputs):
        """
        Update the state of the network
        params:
            state: previous state of the network, shape: (n_size,)
            inputs: input data, shape: (in_features,)
        """
        input_signal = []
        for i in range(self.n_task):
            input_signal.append(self.in_scale * inputs[:, i] @ self.W_ir_list[i].T)
        input_signal = np.array(input_signal).sum(axis=0)
        preactivation = state @ self.W_rc.T + input_signal
        state = (1 - self.lr) * state + self.lr * self.act(preactivation)
        return state
