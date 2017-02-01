import nengo
import numpy as np
import pytry

import cogsci17_decide.networks


class DecisionTrial(pytry.NengoTrial):
    def params(self):
        self.param("network under test", network='LCA')
        self.param("# choices", d=10)
        self.param("# neurons per choice", N=200)

        self.param("input baseline", baseline=0.3)
        self.param("target separation", target_sep=0.2)
        self.param("noise std", noise=0.)
        self.param("input scaling", scale=1.)

        self.param("Share thresholding intercepts in DD network.",
                   share_thresholding_intercepts=False)

    def model(self, p):
        with nengo.Network(seed=p.seed) as model:
            net_args = dict(d=p.d, n_neurons=p.N, dt=p.dt)
            if p.share_thresholding_intercepts:
                net_args['share_thresholding_intercepts'] = True
            decide = getattr(cogsci17_decide.networks, p.network)(**net_args)

            stimulus = p.baseline * np.ones(p.d)
            stimulus[1:] -= p.target_sep
            stimulus_node = nengo.Node(stimulus, label="Input")
            nengo.Connection(stimulus_node, decide.input, synapse=None,
                             transform=p.scale)

            if p.noise > 0.:
                noise_node = nengo.Node(nengo.processes.WhiteNoise(
                    nengo.dists.Gaussian(.0, p.noise)), size_out=p.d,
                    label="Noise")
            else:
                noise_node = nengo.Node(np.zeros(p.d), label="Noise")
            nengo.Connection(noise_node, decide.input, synapse=None,
                             transform=p.scale)

            self.probe = nengo.Probe(decide.output, synapse=0.01)
        return model

    def evaluate(self, p, sim, plt):
        sim.run(2.)

        if plt is not None:
            plt.plot(sim.trange(), sim.data)
            plt.xlabel("Time [s]")

        ss_data = sim.data[self.probe][sim.trange() > 1., :]
        smoothed = np.mean(ss_data, axis=0)
        winner = np.argmax(smoothed)
        mask = np.ones(p.d, dtype=bool)
        mask[winner] = False
        runnerup = np.argmax(smoothed[mask])
        if runnerup >= winner:
            runnerup += 1

        threshold = 0.15
        decided = (
            np.all(ss_data[:, winner] > threshold) and
            np.all(ss_data[:, runnerup] <= threshold))
        correct = decided and np.all(np.argmax(ss_data, axis=1) == 0)
        t = np.nan
        if decided:
            risen = np.flatnonzero(
                sim.data[self.probe][:, winner] > threshold)
            if len(risen) > 0:
                fall = np.flatnonzero(
                    np.max(sim.data[self.probe], axis=1) <= threshold)
                idx = risen[0]
                if np.any(fall > idx):
                    idx = fall[-1]
                if idx >= len(sim.trange() - 1):
                    t = np.nan
                else:
                    t = sim.trange()[idx]

        return dict(
            decided=decided,
            correct=correct,
            t=t,
            runnerup_highest_err=max(0., np.max(sim.data[self.probe][:, mask]))
        )
