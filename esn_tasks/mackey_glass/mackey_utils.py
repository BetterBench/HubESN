import numpy as np

def generate_mackey(N):
    gamma   = 0.1
    beta   = 0.2
    tau = 17

    y = [0.9697, 0.9699, 0.9794, 1.0003, 1.0319, 1.0703, 1.1076, 1.1352, 1.1485,
        1.1482, 1.1383, 1.1234, 1.1072, 1.0928, 1.0820, 1.0756, 1.0739, 1.0759]

    for n in range(17,N+99):
        y.append(y[n] - gamma*y[n] + beta*y[n-tau]/(1+y[n-tau]**10))

    return np.array(y[100:])