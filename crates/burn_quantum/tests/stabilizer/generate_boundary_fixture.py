#!/usr/bin/env python3
"""Independent integer-GF(2) geometry/flow oracle, verified by pinned Stim 1.15.0.

No burn_quantum code is imported or executed. Regenerate with:
  python3 generate_boundary_fixture.py --stim /path/to/stim --output boundary-stim-1.15.0.json
The Stim executable must have been built from the declared pinned source release.
"""
import argparse
import hashlib
import itertools
import json
import subprocess


def echelon(rows):
    pivots = {}
    for value in rows:
        while value:
            pivot = value.bit_length() - 1
            if pivot in pivots:
                value ^= pivots[pivot]
            else:
                pivots[pivot] = value
                break
    return pivots


def reduce(value, basis):
    while value:
        pivot = value.bit_length() - 1
        if pivot not in basis:
            return value
        value ^= basis[pivot]
    return 0


def nullspace(rows, columns):
    pivots = echelon(rows)
    out = []
    for free in range(columns):
        if free in pivots:
            continue
        value = 1 << free
        for pivot in sorted(pivots):
            if (pivots[pivot] & value).bit_count() % 2:
                value |= 1 << pivot
        out.append(value)
    return out


def commutes(a, b):
    return ((a[0] & b[1]).bit_count() + (a[1] & b[0]).bit_count()) % 2 == 0


def multiply(a, b):
    """i^phase X^x Z^z; canonical Hermitian Paulis have phase popcount(x&z)."""
    return a[0] ^ b[0], a[1] ^ b[1], (a[2] + b[2] + 2 * (a[1] & b[0]).bit_count()) % 4


def terms(x, z):
    return [[q, 'Y' if x >> q & 1 and z >> q & 1 else 'X' if x >> q & 1 else 'Z']
            for q in range(max(x.bit_length(), z.bit_length())) if (x | z) >> q & 1]


def geometry(width, depth, window=None):
    upper = 2 * width + 1 if window is not None else width
    cells = [(u, v, w) for u in range(upper + 1) for v in range(width + 1)
             for w in range(depth + 1) if u % 2 == v % 2 == w % 2]
    index = {cell: i for i, cell in enumerate(cells)}
    deltas = [(1, 1, -1), (1, -1, 1), (-1, 1, 1)]
    ports = [1, 2, 0]
    fusions, terminal, outputs = [], [], [[], []]
    for ci, cell in enumerate(cells):
        for axis, delta in enumerate(deltas):
            for sign in [1, -1]:
                port = ports[axis] + (3 if sign < 0 else 0)
                q = 6 * ci + port
                neighbor = tuple(x + sign * y for x, y in zip(cell, delta))
                cut = window is not None and ((cell[0] <= width) != (neighbor[0] <= width))
                enabled = window is None or window[0] <= (cell[2] + neighbor[2]) / 2 < window[1]
                if cut and not enabled:
                    terminal.append((q, 'X'))
                elif neighbor in index:
                    if sign > 0:
                        fusions.append((q, 6 * index[neighbor] + ports[axis] + 3))
                elif not 0 <= neighbor[0] <= upper:
                    terminal.append((q, 'X'))
                elif not 0 <= neighbor[1] <= width:
                    terminal.append((q, 'Z'))
                else:
                    outputs[int(neighbor[2] > depth)].append(q)
    # Match the explicit stream contract: all XX/ZZ fusion records, then terminals.
    measurements = [p for a, b in fusions for p in [(1 << a | 1 << b, 0), (0, 1 << a | 1 << b)]]
    measurements += [(1 << q, 0) if p == 'X' else (0, 1 << q) for q, p in terminal]
    stabilizers = [(1 << (6 * ci + port), (1 << (6 * ci + (port + 1) % 6)) |
                   (1 << (6 * ci + (port - 1) % 6))) for ci in range(len(cells)) for port in range(6)]
    consumed = [q for pair in fusions for q in pair] + [q for q, _ in terminal]
    assert sorted(consumed + outputs[0] + outputs[1]) == list(range(6 * len(cells)))
    return cells, fusions, terminal, outputs, measurements, stabilizers


def relations(output, measurements, stabilizers):
    observables = measurements + [(1 << q, 0) for q in output] + [(0, 1 << q) for q in output]
    constraints = [sum((not commutes(s, p)) << i for i, p in enumerate(observables)) for s in stabilizers]
    return nullspace(constraints, len(observables))


def signed_relation(row, output, measurements, stabilizers):
    m, n = len(measurements), len(output)
    coefficients = [i for i in range(m) if row >> i & 1]
    x = sum(1 << q for j, q in enumerate(output) if row >> (m + j) & 1)
    z = sum(1 << q for j, q in enumerate(output) if row >> (m + n + j) & 1)
    measured = (0, 0, 0)
    for i in coefficients:
        a, b = measurements[i]
        measured = multiply(measured, (a, b, (a & b).bit_count() % 4))
    product = multiply(measured, (x, z, (x & z).bit_count() % 4))
    resource = (0, 0, 0)
    for q, (a, b) in enumerate(stabilizers):
        if product[0] >> q & 1:
            resource = multiply(resource, (a, b, 0))
    assert resource[:2] == product[:2]
    phase = (product[2] - resource[2]) % 4
    assert phase in [0, 2]
    return {'terms': terms(x, z), 'measurement_indices': coefficients, 'negative': phase == 2}


def local_code(output, measurements, stabilizers):
    n, m = len(output), len(measurements)
    rows = [row >> m for row in relations(output, measurements, stabilizers) if row >> m]
    basis = list(echelon(rows).values())
    mask = (1 << n) - 1
    # Derive CSS subgroups by linear elimination, not by assuming the returned basis is CSS.
    def subgroup(x_only):
        forbidden = [(row >> n) if x_only else (row & mask) for row in basis]
        constraints = [sum(((row >> q) & 1) << i for i, row in enumerate(forbidden)) for q in range(n)]
        combinations = nullspace(constraints, len(basis))
        result = []
        for bits in combinations:
            row = 0
            for i, value in enumerate(basis):
                if bits >> i & 1:
                    row ^= value
            result.append((row & mask) if x_only else row >> n)
        return list(echelon(result).values())
    xs, zs = subgroup(True), subgroup(False)
    assert len(xs) + len(zs) == len(basis), 'end code must be CSS for this fixture'
    logicals = []
    for own, other, pauli in [(xs, zs, 'X'), (zs, xs, 'Z')]:
        own_basis = echelon(own)
        candidates = [v for v in nullspace(other, n) if reduce(v, own_basis)]
        if candidates:
            value = min(candidates, key=int.bit_count)
            logicals.append([[output[i], pauli] for i in range(n) if value >> i & 1])
    return {'outputs': output, 'rank': len(basis), 'logical_qubits': n - len(basis),
            'x_checks': [[output[i] for i in range(n) if row >> i & 1] for row in xs],
            'z_checks': [[output[i] for i in range(n) if row >> i & 1] for row in zs],
            'logical_representatives': logicals}


def case(width, depth, window, stim, seed):
    cells, fusions, terminal, ends, measurements, stabilizers = geometry(width, depth, window)
    output = ends[0] + ends[1]
    m = len(measurements)
    generators, basis = [], {}
    checks = [signed_relation(row, [], measurements, stabilizers)
              for row in relations([], measurements, stabilizers)]
    for row in relations(output, measurements, stabilizers):
        value = row >> m
        if value and reduce(value, basis):
            generators.append(signed_relation(row, output, measurements, stabilizers))
            basis = echelon([*basis.values(), value])
    assert len(generators) == len(output), 'conditional output must be pure'
    circuit = ['H ' + ' '.join(map(str, range(6 * len(cells))))]
    circuit += [f'CZ {6*c+q} {6*c+(q+1)%6}' for c in range(len(cells)) for q in range(6)]
    all_terms = [terms(x, z) for x, z in measurements] + [g['terms'] for g in generators]
    circuit += ['MPP ' + '*'.join(f'{p}{q}' for q, p in value) for value in all_terms]
    source = '\n'.join(circuit) + '\n'
    result = subprocess.run([stim, 'sample', '--shots', '32', '--seed', str(seed)],
                            input=source, text=True, capture_output=True, check=True)
    samples = result.stdout.splitlines()
    assert len(samples) == 32 and all(len(v) == len(all_terms) for v in samples)
    for bits in samples:
        for i, relation in enumerate(generators):
            parity = sum(bits[j] == '1' for j in relation['measurement_indices']) % 2
            assert (parity ^ (bits[m+i] == '1')) == relation['negative']
        for relation in checks:
            assert (sum(bits[j] == '1' for j in relation['measurement_indices']) % 2) == relation['negative']
    local = [local_code(end, measurements, stabilizers) for end in ends]
    groups = [[q for q in end if cells[q//6][0] <= width] for end in ends]
    if window is not None:
        groups = [group for end in ends for group in
                  [[q for q in end if cells[q//6][0] <= width], [q for q in end if cells[q//6][0] > width]]]
    return {'width': width, 'depth': depth, 'merge_window': window, 'cells': cells,
            'fusions': fusions, 'terminal': terminal, 'ends': ends, 'measurement_count': m,
            'output_relations': generators, 'checks': checks, 'end_codes': local,
            'patch_codes': [local_code(group, measurements, stabilizers) for group in groups],
            'transmitted_logicals': local[0]['logical_qubits'],
            'stim_seed': seed, 'stim_shots': 32, 'stim_circuit_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'stim_records': samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stim', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cases = [case(width, 3, None, args.stim, 610 + width) for width in [3, 5, 7]]
    cases += [case(3, 10, window, args.stim, 630 + i) for i, window in enumerate([(0, 0), (1, 4), (2, 4)])]
    # Odd temporal extent retains both full end codes; even depth10 above is a
    # deliberate channel-only control with a shortened output boundary code.
    cases += [case(3, 11, (1, 4), args.stim, 640), case(5, 13, (1, 6), args.stim, 641)]
    data = {'stim_version': '1.15.0', 'generator': 'independent integer symplectic algebra and Stim circuit sampling',
            'cases': cases}
    with open(args.output, 'w') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
    print([(c['width'], c['merge_window'], c['transmitted_logicals']) for c in cases])


if __name__ == '__main__':
    main()
