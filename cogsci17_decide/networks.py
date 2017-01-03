import nengo
import numpy as np


def UsherMcClelland(d, n_neurons, dt):
    k = 1.
    beta = 1.
    tau_model = 0.1

    tau_actual = 0.1
    a = np.exp(-dt / tau_actual)

    # eqn (4) ignoring truncation, put into the canonical form:
    #   x[t+dt] = Ax[t] + Bu
    inhibit = np.ones((d, d))
    inhibit[np.diag_indices(d)] = 0.
    I = np.eye(d)
    B = dt / tau_model
    A = (-k * I - beta * inhibit) * dt / tau_model + I

    with nengo.Network() as net:
        net.input = nengo.Node(size_in=d)
        x = nengo.networks.EnsembleArray(n_neurons, d)

        nengo.Connection(
            net.input, x.input,
            transform=B / (1 - a),  # discrete principle 3
            synapse=tau_actual)

        nengo.Connection(
            x.output, x.input,
            transform=(A - a * I) / (1 - a),  # discrete principle 3
            synapse=tau_actual)

        net.output = x.output

    return net


def DriftDiffusion(d, n_neurons, dt, share_thresholding_intercepts=False):
    k = 0.
    beta = 0.
    tau_model = 0.1

    tau_actual = 0.1
    a = np.exp(-dt / tau_actual)

    # eqn (4) ignoring truncation, put into the canonical form:
    #   x[t+dt] = Ax[t] + Bu
    inhibit = np.ones((d, d))
    inhibit[np.diag_indices(d)] = 0.
    I = np.eye(d)
    B = 1.0 * dt / tau_model
    A = (-k * I - beta * inhibit) * dt / tau_model + I

    n_neurons_threshold = 50
    n_neurons_x = n_neurons - n_neurons_threshold
    assert n_neurons_x > 0
    threshold = 0.8

    print(B / (1 - a))
    print((A - a * I) / (1 - a))
    with nengo.Network() as net:
        net.input = nengo.Node(size_in=d)
        x = nengo.networks.EnsembleArray(n_neurons_x, d)

        nengo.Connection(
            net.input, x.input,
            transform=B / (1 - a),  # discrete principle 3
            synapse=tau_actual)

        nengo.Connection(
            x.output, x.input,
            transform=(A - a * I) / (1 - a),  # discrete principle 3
            synapse=tau_actual)

        with nengo.presets.ThresholdingEnsembles(0.):
            thresholding = nengo.networks.EnsembleArray(n_neurons_threshold, d)
            if share_thresholding_intercepts:
                for e in thresholding.ensembles:
                    e.intercepts = nengo.dists.Exponential(
                        0.15, 0., 1.).sample(n_neurons_threshold)
            net.output = thresholding.add_output('heaviside', lambda x: x > 0.)

        bias = nengo.Node(1.)

        nengo.Connection(x.output, thresholding.input)
        nengo.Connection(
            bias, thresholding.input, transform=-threshold * np.ones((d, 1)))
        nengo.Connection(
            thresholding.heaviside, x.input,
            transform=-2. + 3. * np.eye(d), synapse=tau_actual)

    return net
