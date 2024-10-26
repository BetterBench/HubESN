# model params
params = {
    "p2": 0.3,
    "lambda_dc": 0.6, # distance constraint
    "lambda_sc": 0.4, # sequence constraint
    "lambda_hc": 0, # hebbian constraint
    "input_neurons": "hub",
    "nonhub_type": "out",
    "r_sig": 0.4,
    "n_size": 300,
    "in_features": 6,
    "in_scale": 1e-5,
    "activation": "tanh",
    "spec_rad": 1,
    "lr": 1,
    "verbose": False,
}

# task params
task_pm = {
    'n_bit': 4,
    'bit_len': 5,
    'T0': 50,
    'n_trials': -1,
    'test_ratio': -1,
    'shuffle': True
}

# search space
p2: [0.1, 0.2, 0.3, 0.4]
r_sig: [0.1, 0.2, 0.3, 0.4, 0.5]
in_scale: [1e-4, 1e-5, 1e-6]
lambda_dc: [0, 0.2, 0.4, 0.6]