# A Spiking Independent Accumulator Model for Winner-Take-All Computation
Winner-take-all (WTA) mechanisms are an important component of many cognitive models.
For example, they are often used to decide between multiple choices or to selectively direct attention.
Here we compare two biologically plausible, spiking neural WTA mechanisms. 
We first provide a novel spiking implementation of the well-known leaky, competing accumulator (LCA) model, by mapping the dynamics onto a population-level representation.
We then propose a two-layer spiking independent accumulator (IA) model, and compare its performance against the LCA network on a variety of WTA benchmarks.
Our findings suggest that while the LCA network can rapidly adapt to new winners, the IA network is better suited for stable decision making in the presence of noise.

## Important files

### Paper
- See `doc/paper` for the uncompiled LaTeX paper.
- See `doc/supplementary/supplementary.pdf` for the supplementary analysis PDF.

### Source code
- See `cogsci17_decide/networks.py` for the network implementations.
- See `cogsci17_decide/trials.py` for the benchmark implementations.

### Data files
- Data presented in the paper is stored in the file `data/decision.npz`.
- Data with number of choices increased to 20 is stored in the file
  `data/more_d.npz`

Data files can be loaded with `numpy.load`.

### Data analysis and plots
The `notebooks` directory contains [Jupyter notebooks](http://jupyter.org/) with
source code for data analysis and plotting.

## Running simulations

1. Clone this repository: `git clone https://github.com/ctn-waterloo/cogsci17-decide.git`
2. Install as Python package: `cd cogsci17-decide && pip install .`

### Running single simulations

Single simulations can be run by invoking `pytry cogsci_decide/trial.py`. There
are various command line arguments to set network parameters. For example,
`pytry cogsci_decide/trial.py --network IA` runs the independent accumulator
network instead of the leaky, competing accumulator model. A full list of
possible arguments can be printed with `pytry cogsci_decide/trial.py --help`.

The networks can be run in Nengo GUI to inspect and plot different neural
populations with the `--gui` option. Note that this requires
[Nengo GUI](https://github.com/nengo/nengo_gui) to be installed.

### Running batch simulations

1. To run simulations in batch to collect data, install [Psyrun (commit
659307a3)](https://github.com/jgosmann/psyrun/tree/65307a3d452b3eab1cf1335de14217be01e0a825).
2. `psy-doit decision` will generate the data presented in the paper; `psy-doit
   more_d` will generate data with 20 dimensions instead of 10. Use the `-n
<number>` flag to specify how many CPU cores to use. This can cut down the
simulation time significantly (still expect a few hours on a modern computer).
3. Afterwards, data can be found in `psy-work/decision/result.npz` or
   `psy-work/more_d/result.npz`.

## Dependencies

Version numbers state the versions used to generate the data, newer and older
versions might work as well, but have not been tested.

### General
- [Python 3.5.2](https://www.python.org/)
- [Nengo 2.3.0](https://github.com/nengo/nengo)
- [Numpy 1.12.0](http://www.numpy.org/)
- [PyTry 0.9.1](https://github.com/tcstewar/pytry)

### Jupyter notebooks
- [Jupyter](http://jupyter.org/)
  - jupyter 1.0.0
  - jupyter-client 4.4.0
  - jupyter-core 4.2.1
  - notebook 4.2.3
- [Matplotlib 1.5.2](http://matplotlib.org/)
- [Pandas 0.18.1](http://pandas.pydata.org/)
- [Seaborn 0.7.1](http://seaborn.pydata.org/)

### Live visualization
- [Nengo GUI 0.2](https://github.com/nengo/nengo_gui)

### Batch simulations
- [Psyrun @659307a3](https://github.com/jgosmann/psyrun/tree/65307a3d452b3eab1cf1335de14217be01e0a825).
