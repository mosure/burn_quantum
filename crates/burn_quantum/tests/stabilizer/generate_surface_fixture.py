#!/usr/bin/env python3
"""Independent axis-boundary six-ring flows, sampled with pinned Stim 1.15.0.

No burn_quantum code is imported. The earlier staircase/BCC fixture is unchanged.
Records are all fusion XX/ZZ pairs, then terminal side measurements. Canonical
logical operators include their physical Pauli lift sign, including encoded Y.
Run regenerate_stim.py with the pinned source archive to rebuild the CLI and fixture.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from generate_boundary_fixture import (
    commutes, echelon, multiply, nullspace, reduce, relations, signed_relation, terms,
)


def corner_factor(axis, positive):
    corner = [0, 0, 0]
    corner[axis] = int(positive)
    vertices = [p + (3 if c == 0 else 0) for p, c in zip([1, 2, 0], corner)]
    if len({v % 2 for v in vertices}) == 1:
        return 'X', corner
    center = next(v for v in vertices if (v+1) % 6 in vertices and (v-1) % 6 in vertices)
    port = [1, 2, 0][axis] + (0 if positive else 3)
    return ('X' if port == center else 'Z'), corner


def geometry(size, depth, window):
    paired = window is not None
    cells = [(x, y, z) for x in range(size * (2 if paired else 1))
             for y in range(size) for z in range(depth)]
    index = {c: i for i, c in enumerate(cells)}
    fusions, terminal = [], []
    ends = [[[] for _ in range(2 if paired else 1)] for _ in range(2)]
    for ci, c in enumerate(cells):
        for axis, port in enumerate([1, 2, 0]):
            for sign in [1, -1]:
                neighbor = list(c)
                neighbor[axis] += sign
                q = 6*ci + port + (3 if sign < 0 else 0)
                seam = paired and axis == 0 and ((c[0] < size) != (neighbor[0] < size))
                cut = seam and not (window[0] <= c[2] < window[1])
                if tuple(neighbor) in index and not cut:
                    if sign > 0:
                        fusions.append([q, 6*index[tuple(neighbor)] + port + 3])
                elif axis == 2:
                    ends[int(sign > 0)][int(paired and c[0] >= size)].append(q)
                else:
                    factor, corner = corner_factor(axis, sign > 0)
                    keep_color = 1 if axis == 0 else 0
                    color = (sum(c) + sum(corner)) % 2
                    basis = factor if color == keep_color else ('Z' if factor == 'X' else 'X')
                    terminal.append([q, basis])
    groups = [sorted(group) for end in ends for group in end]
    measured = [p for a, b in fusions for p in [(1 << a | 1 << b, 0), (0, 1 << a | 1 << b)]]
    measured += [(1 << q, 0) if p == 'X' else (0, 1 << q) for q, p in terminal]
    stabilizers = [(1 << (6*c+q), (1 << (6*c+(q+1) % 6)) | (1 << (6*c+(q-1) % 6)))
                   for c in range(len(cells)) for q in range(6)]
    consumed = [q for pair in fusions for q in pair] + [q for q, _ in terminal]
    assert sorted(consumed + sum(groups, [])) == list(range(6*len(cells)))
    return cells, fusions, terminal, groups, measured, stabilizers


def frame_mask(x, z, frame):
    swap = (x ^ z) & frame
    return x ^ swap, z ^ swap


def code(group, cells, measured, stabilizers):
    n, m = len(group), len(measured)
    mask = (1 << n) - 1
    hadamards = [q for q in group if sum(cells[q//6][:2]) % 2]
    frame = sum(1 << i for i, q in enumerate(group) if q in hadamards)
    rows = []
    for row in relations(group, measured, stabilizers):
        row >>= m
        if row:
            x, z = frame_mask(row & mask, row >> n, frame)
            rows.append(x | (z << n))
    basis = list(echelon(rows).values())
    pure = []
    for axis in range(2):
        forbidden = [(v >> n) if axis == 0 else v & mask for v in basis]
        constraints = [sum(((v >> q) & 1) << j for j, v in enumerate(forbidden)) for q in range(n)]
        generators = []
        for combination in nullspace(constraints, len(basis)):
            value = 0
            for i, row in enumerate(basis):
                if combination >> i & 1:
                    value ^= row
            generators.append((value & mask) if axis == 0 else value >> n)
        pure.append(list(echelon(generators).values()))
    assert len(pure[0]) + len(pure[1]) == len(basis), 'local-H frame must make the end code CSS'
    assert len(basis) == n-1, 'each spatial end block must encode one qubit'
    representatives = []
    for axis in range(2):
        own = echelon(pure[axis])
        value = min((v for v in nullspace(pure[1-axis], n) if reduce(v, own)), key=lambda v: (v.bit_count(), v))
        x, z = frame_mask(value if axis == 0 else 0, value if axis == 1 else 0, frame)
        representatives.append((sum(1 << q for i, q in enumerate(group) if x >> i & 1),
                                sum(1 << q for i, q in enumerate(group) if z >> i & 1)))
    assert not commutes(*representatives)
    return {'photons': group, 'hadamards': hadamards, 'rank': len(basis), 'logical_qubits': 1,
            'css_x_checks': [[group[i] for i in range(n) if v >> i & 1] for v in pure[0]],
            'css_z_checks': [[group[i] for i in range(n) if v >> i & 1] for v in pure[1]],
            'logical_x': terms(*representatives[0]), 'logical_z': terms(*representatives[1])}, representatives


def physical_vector(row, output, m):
    n = len(output)
    return (sum(1 << q for i, q in enumerate(output) if row >> (m+i) & 1),
            sum(1 << q for i, q in enumerate(output) if row >> (m+n+i) & 1))


def signed_target(target, output, measured, stabilizers, pivots):
    n, m = len(output), len(measured)
    value = sum(((target[0] >> q) & 1) << i for i, q in enumerate(output))
    value |= sum(((target[1] >> q) & 1) << (n+i) for i, q in enumerate(output))
    row = 0
    while value:
        pivot = value.bit_length()-1
        vector, previous = pivots[pivot]
        value ^= vector
        row ^= previous
    assert physical_vector(row, output, m) == target
    return signed_relation(row, output, measured, stabilizers)


def lift(vector, reps):
    product = (0, 0, 0)
    encoded = []
    for i, (lx, lz) in enumerate(reps):
        x, z = (vector >> (2*i)) & 1, (vector >> (2*i+1)) & 1
        if x:
            product = multiply(product, (*lx, (lx[0] & lx[1]).bit_count() % 4))
        if z:
            product = multiply(product, (*lz, (lz[0] & lz[1]).bit_count() % 4))
        if x and z:
            product = (product[0], product[1], (product[2]+1) % 4)
        if x or z:
            encoded.append([i, 'Y' if x and z else 'X' if x else 'Z'])
    lift_phase = (product[2]-(product[0] & product[1]).bit_count()) % 4
    assert lift_phase in [0, 2]
    return product[:2], encoded, lift_phase == 2


def case(size, depth, window, stim, seed):
    cells, fusions, terminal, groups, measured, stabilizers = geometry(size, depth, window)
    output, m = sum(groups, []), len(measured)
    all_rows = relations(output, measured, stabilizers)
    checks = [signed_relation(row, [], measured, stabilizers) for row in relations([], measured, stabilizers)]
    pivots = {}
    generators = []
    for row in all_rows:
        value, original = row >> m, row
        while value:
            pivot = value.bit_length()-1
            if pivot not in pivots:
                pivots[pivot] = (value, row)
                generators.append(signed_relation(original, output, measured, stabilizers))
                break
            v, r = pivots[pivot]
            value ^= v
            row ^= r
    assert len(generators) == len(output)
    codes, reps = zip(*(code(g, cells, measured, stabilizers) for g in groups))
    logical_vectors, logical_basis = [], {}
    for row in all_rows:
        physical = physical_vector(row, output, m)
        vector = sum((not commutes(physical, pair[1])) << (2*i) |
                     (not commutes(physical, pair[0])) << (2*i+1) for i, pair in enumerate(reps))
        if not reduce(vector, logical_basis):
            continue
        logical_basis = echelon([*logical_basis.values(), vector])
        logical_vectors.append(vector)
    def logical_relation(vector):
        physical, encoded, negative = lift(vector, reps)
        relation = signed_target(physical, output, measured, stabilizers, pivots)
        return {**relation, 'encoded_terms': encoded, 'lift_negative': negative,
                'encoded_negative': relation['negative'] ^ negative}
    logicals = [logical_relation(v) for v in logical_vectors]
    products = []
    for combination in range(1, 1 << len(logical_vectors)):
        vector = 0
        for i, value in enumerate(logical_vectors):
            if combination >> i & 1:
                vector ^= value
        products.append(logical_relation(vector))
    assert len(logicals) == len(groups)
    circuit = ['H ' + ' '.join(map(str, range(6*len(cells))))]
    circuit += [f'CZ {6*c+q} {6*c+(q+1)%6}' for c in range(len(cells)) for q in range(6)]
    targets = generators + logicals + products
    all_terms = [terms(x, z) for x, z in measured] + [g['terms'] for g in targets]
    circuit += ['MPP ' + '*'.join(f'{p}{q}' for q, p in value) for value in all_terms]
    source = '\n'.join(circuit) + '\n'
    result = subprocess.run([stim, 'sample', '--shots', '32', '--seed', str(seed)],
                            input=source, text=True, capture_output=True, check=True)
    samples = result.stdout.splitlines()
    assert len(samples) == 32 and all(len(v) == len(all_terms) for v in samples)
    for bits in samples:
        for i, relation in enumerate(targets):
            parity = sum(bits[j] == '1' for j in relation['measurement_indices']) % 2
            assert (parity ^ (bits[m+i] == '1')) == relation['negative']
        for relation in checks:
            assert (sum(bits[j] == '1' for j in relation['measurement_indices']) % 2) == relation['negative']
    left = sum(groups[:len(groups)//2], [])
    local_rank = len(echelon([v >> m for v in relations(left, measured, stabilizers)]))
    transmitted = len(left)-local_rank
    assert transmitted == (2 if window == (0, 0) else 1)
    return {'distance_parameter': size, 'depth': depth, 'merge_window': window,
            'cells': cells, 'fusions': fusions, 'terminal': terminal, 'end_blocks': groups,
            'measurement_count': m, 'output_relations': generators, 'checks': checks,
            'end_codes': codes, 'logical_relations': logicals, 'logical_product_relations': products,
            'transmitted_logicals': transmitted,
            'stim_seed': seed, 'stim_shots': 32,
            'stim_circuit_sha256': hashlib.sha256(source.encode()).hexdigest(), 'stim_records': samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stim', required=True)
    parser.add_argument('--source-archive', required=True,
                        help='Pinned Stim 1.15.0 source archive used to build this executable')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source_pin = json.loads((Path(__file__).parent/'stim-1.15.0.json').read_text())['source_archive_sha256']
    assert hashlib.sha256(Path(args.source_archive).read_bytes()).hexdigest() == source_pin, 'source archive hash mismatch'
    # Explicit phase guards beyond the actual geometry: i XXX ZZZ = -YYY,
    # while H on one overlapping site reverses this sign.
    assert lift(3, [((7, 0), (0, 7))])[2]
    assert not lift(3, [((6, 1), (1, 6))])[2]
    cases = []
    for i, (size, depth, window) in enumerate([(3, 3, None), (5, 5, None),
                                               (3, 5, (0, 0)), (3, 5, (1, 4)), (3, 5, (2, 3))]):
        cases.append(case(size, depth, window, args.stim, 810+i))
        print('verified', size, depth, window, flush=True)
    data = {'stim_version': '1.15.0', 'source_archive_sha256': source_pin,
            'generator': 'independent axis geometry and signed integer symplectic algebra with Stim sampling',
            'record_order': 'fusion XX/ZZ pairs then terminal side measurements; output_relations then logical_relations then logical_product_relations physical masks appended',
            'boundary_colors': {'x': 1, 'y': 0}, 'cases': cases}
    Path(args.output).write_text(json.dumps(data, indent=2)+'\n')


if __name__ == '__main__':
    main()
