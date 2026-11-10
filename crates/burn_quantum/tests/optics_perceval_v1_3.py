"""Regenerate pinned independent optical fixtures; run in a disposable environment.

python -m pip install perceval-quandela==1.3.0 exqalibur==1.4.2 numpy==2.5.4
MPLCONFIGDIR=/tmp/burn-quantum-mpl python optics_perceval_v1_3.py

The simulator exports unconditional loss probabilities (no photon-count filter).
"""
import importlib.metadata
import json
import math
from pathlib import Path
import numpy as np
import perceval as pcvl

assert pcvl.__version__ == '1.3.0'
assert importlib.metadata.version('exqalibur') == '1.4.2'
assert np.__version__ == '2.5.4'


def occupations(modes, total):
    if modes == 1:
        return [[total]]
    return [[n, *rest] for n in range(total + 1)
            for rest in occupations(modes - 1, total - n)]


matrix = np.array([[np.exp(2j * math.pi * i * j / 3) / math.sqrt(3)
                    for j in range(3)] for i in range(3)])
backend = pcvl.backends.BackendFactory.get_backend('SLOS')
backend.set_circuit(pcvl.Circuit(3).add(0, pcvl.Unitary(pcvl.Matrix(matrix))))
basis = occupations(3, 3)
amplitudes = [[None] * len(basis) for _ in basis]
for column, incoming in enumerate(basis):
    backend.set_input_state(pcvl.BasicState(incoming))
    for row, outgoing in enumerate(basis):
        value = backend.prob_amplitude(pcvl.BasicState(outgoing))
        amplitudes[row][column] = [value.real, value.imag]

experiment = pcvl.Experiment(2)
experiment.add(0, pcvl.BS.Ry())
experiment.add(0, pcvl.LC(0.2))
experiment.add(1, pcvl.LC(0.6))
simulator = pcvl.SimulatorFactory.build(experiment, backend='SLOS')
probabilities = simulator.probs(pcvl.BasicState([1, 1]))
loss_basis = [o for n in range(3) for o in occupations(2, n)]
loss_probabilities = [float(probabilities.get(pcvl.BasicState(o), 0.)) for o in loss_basis]
assert abs(sum(loss_probabilities) - 1.) < 1e-12
output = {
    'perceval': pcvl.__version__,
    'exqalibur': importlib.metadata.version('exqalibur'),
    'numpy': np.__version__,
    'backend': 'SLOS',
    'convention': 'single-photon matrix rows=output, columns=input; Ry splitter; unfiltered probabilities',
    'fourier_3_modes_3_photons': {'basis': basis, 'amplitudes_output_input': amplitudes},
    'hom_asymmetric_loss': {'transmissions': [0.8, 0.4], 'basis': loss_basis, 'probabilities': loss_probabilities},
}
path = Path(__file__).with_suffix('.json')
path.write_text(json.dumps(output, indent=2) + '\n')
print(path)
