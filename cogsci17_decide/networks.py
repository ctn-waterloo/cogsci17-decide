import nengo
import numpy as np


def UsherMcClelland(d, n_neurons, dt):
    k = 1.
    beta = 1.
    tau_model = 0.01

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
        x = nengo.Ensemble(d * n_neurons, d)

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


def DriftDiffusion(d, n_neurons, dt):
    n_neurons = n_neurons // 2  # two-layered network

    rec_synapse = 0.1
    threshold = 0.8

    with nengo.Network() as net:
        with nengo.presets.ThresholdingEnsembles(0.):
            evidence_accum = nengo.networks.EnsembleArray(n_neurons, d)
            nengo.Connection(
                evidence_accum.output, evidence_accum.input,
                synapse=rec_synapse)
            thresholding = nengo.networks.EnsembleArray(n_neurons, d)
            thresholding.add_output('heaviside', lambda x: x > 0.)

        bias = nengo.Node(1.)

        net.input = nengo.Node(size_in=d)
        net.output = nengo.Node(size_in=d)

        nengo.Connection(
            net.input, evidence_accum.input, transform=0.5,
            synapse=rec_synapse)
        nengo.Connection(evidence_accum.output, thresholding.input)
        nengo.Connection(
            bias, thresholding.input, transform=-threshold * np.ones((d, 1)))
        nengo.Connection(
            thresholding.heaviside, evidence_accum.input,
            transform=-2. + 2.5 * np.eye(d), synapse=rec_synapse)
        nengo.Connection(thresholding.heaviside, net.output, synapse=None)

    return net
