"""Independent pinned SLOS fixture for the first Grice boost and partial HOM.

Use the same isolated environment as optics_perceval_v1_3.py.
Grice spatial unitary: Wein et al., https://arxiv.org/html/1509.00088v2 Eq. (3).
Ancilla is the explicitly supplied Phi+ pair; all 330 four-photon records retained.
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

def backend(matrix):
    result = pcvl.backends.BackendFactory.get_backend('SLOS')
    result.set_circuit(pcvl.Circuit(len(matrix)).add(0, pcvl.Unitary(pcvl.Matrix(matrix))))
    return result

spatial = np.array([[1, 1j, 1j, -1], [1j, 1, -1, 1j],
                    [1j, -1, 1, 1j], [-1, 1j, 1j, 1]]) / 2
grice = backend(np.kron(spatial, np.eye(2)))
basis = occupations(8, 4)
computational = np.zeros((4, len(basis)), dtype=complex)
for a in range(2):
    for b in range(2):
        for rail in range(2):
            incoming = [0] * 8
            for mode in [a, 2+b, 4+rail, 6+rail]:
                incoming[mode] += 1
            grice.set_input_state(pcvl.BasicState(incoming))
            computational[2*a+b] += np.array([
                grice.prob_amplitude(pcvl.BasicState(o)) for o in basis]) / math.sqrt(2)
bell = np.array([(computational[1]+computational[2])/math.sqrt(2),
                 (computational[1]-computational[2])/math.sqrt(2),
                 (computational[0]+computational[3])/math.sqrt(2),
                 (computational[0]-computational[3])/math.sqrt(2)])
probabilities = abs(bell)**2
assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-12)
supported = probabilities > 1e-12
unique = supported.sum(axis=0) == 1
success = probabilities[:, unique].sum(axis=1)
assert np.allclose(success, [1, 1, .5, .5], atol=1e-12)

splitter = np.array([[1, -1], [1, 1]]) / math.sqrt(2)
hom = backend(np.kron(splitter, np.eye(2)))
fine = occupations(4, 2)
columns = []
for incoming in [[1, 0, 1, 0], [1, 0, 0, 1]]:
    hom.set_input_state(pcvl.BasicState(incoming))
    columns.append(np.array([hom.prob_amplitude(pcvl.BasicState(o)) for o in fine]))
partial = []
for visibility in [0., .3, .8, 1.]:
    amplitudes = math.sqrt(visibility)*columns[0] + math.sqrt(1-visibility)*columns[1]
    coarse = [0., 0., 0.]
    for o, p in zip(fine, abs(amplitudes)**2):
        coarse[sum(o[:2])] += float(p)
    partial.append({'visibility': visibility, 'probabilities_02_11_20': coarse})

ancilla_basis = occupations(8, 2)
ancilla_amplitude = np.zeros(len(ancilla_basis), dtype=complex)
for rail in range(2):
    incoming = [0] * 8
    incoming[4+rail] = incoming[6+rail] = 1
    grice.set_input_state(pcvl.BasicState(incoming))
    ancilla_amplitude += np.array([grice.prob_amplitude(pcvl.BasicState(o)) for o in ancilla_basis]) / math.sqrt(2)
ancilla_probabilities = abs(ancilla_amplitude)**2
assert abs(ancilla_probabilities.sum()-1) < 1e-12

output = {'grice_vacuum_signals': {'basis': ancilla_basis, 'probabilities': ancilla_probabilities.tolist()}, 'perceval': pcvl.__version__, 'exqalibur': importlib.metadata.version('exqalibur'),
          'numpy': np.__version__, 'backend': 'SLOS',
          'grice': {'basis': basis, 'bell_order': ['psi+', 'psi-', 'phi+', 'phi-'],
                    'probabilities': probabilities.tolist(), 'resolved_probabilities': success.tolist()},
          'partial_hom': partial}
path = Path(__file__).with_suffix('.json')
path.write_text(json.dumps(output, indent=2) + '\n')
print(f'{path}: grice={success.tolist()}, partial_HOM={partial}')
