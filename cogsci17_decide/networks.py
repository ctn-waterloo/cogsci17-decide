import nengo
import numpy as np


def UsherMcClelland(d, n_neurons, dt):
    k = 1.
    beta = 1.
    tau_model = 0.01

    tau_actual = 0.1
    a = np.exp(-dt / tau_actual)

    # eqn (4) ignoring trunctaion, put into the canonical form:
    #   x[t+dt] = Ax[t] + Bu
    inhibit = np.ones((d, d))
    inhibit[np.diag_indices(d)] = 0.
    I = np.eye(d)
    B = dt / tau_model
    A = (-k * I - beta * inhibit) * dt / tau_model + I

    with nengo.Network() as net:
        net.input = nengo.Node(size_in=d)
        x = nengo.Ensemble(n_neurons, d)

        for i in range(d):
            nengo.Connection(
                net.input[i], x[i],
                transform=B / (1 - a),  # discrete principle 3
                synapse=tau_actual)

        nengo.Connection(
            x, x, transform=(A - a * I) / (1 - a),  # discrete principle 3
            synapse=tau_actual)

        net.output = x

    return net
