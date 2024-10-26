from ZhaozeWang.HubESN.esn_lib.hubesn import HubESN
import numpy as np


params = {
    'lr': 0.5,
    'p2': 0.2,
    'r_sig': 0.1,
    'n_size': 500,
    'spec_rad': 0.1,
    'in_features': 3,
    'in_scale': 1,
    'activation': 'tanh',
    'input_neurons': 'rand',
    'hub_type': 'mix',
    'use_skip': False,
    'verbose': False,
    'lambda_dc': 0.5,
    'lambda_sc': 0.5,
    'exp_coef': 2,
}

esn = HubESN(**params)
esn.plot_eigenvalues()