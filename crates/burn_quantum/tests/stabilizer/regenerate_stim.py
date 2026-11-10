#!/usr/bin/env python3
"""Regenerate the pinned Stim 1.15.0 oracle without installing Python bindings.

Usage: python3 regenerate_stim.py /path/to/stim-1.15.0.tar.gz
Fetch the archive separately with: python3 -m pip download --no-deps --no-binary=stim stim==1.15.0
Requires a C++20 compiler; uses a temporary build directory and 8 compile workers.
"""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

folder = Path(__file__).resolve().parent
fixture = folder / 'stim-1.15.0.json'
expected = json.loads(fixture.read_text())['source_archive_sha256']
archive = Path(sys.argv[1])
assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected, 'source archive hash mismatch'
with tempfile.TemporaryDirectory(prefix='burn-quantum-stim-') as temp:
    temp = Path(temp)
    with tarfile.open(archive) as source:
        source.extractall(temp, filter='data')
    root = temp / 'stim-1.15.0'
    sources = [p for p in root.glob('src/**/*.cc')
               if p.name != 'main.cc' and not any(p.name.endswith(s) for s in ['.pybind.cc', '.test.cc', '.perf.cc'])]
    def build(source):
        obj = temp / (str(source.relative_to(root)).replace('/', '_') + '.o')
        subprocess.run(['g++', '-std=c++20', '-O1', '-g0', '-fno-strict-aliasing',
                        '-DVERSION_INFO=1.15.0', '-I'+str(root/'src'), '-c', str(source), '-o', str(obj)], check=True)
        return str(obj)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        objects = list(pool.map(build, sources))
    for generator, output in [
        ('generate_stim_fixture.cc', 'stim-1.15.0.json'),
        ('generate_six_ring_fixture.cc', 'six-ring-stim-1.15.0.json'),
        ('generate_surgery_fixture.cc', 'surgery-stim-1.15.0.json'),
    ]:
        program = temp / 'oracle'
        subprocess.run(['g++', '-std=c++20', '-O1', '-I'+str(root/'src'),
                        str(folder/generator), *objects, '-pthread', '-o', str(program)], check=True)
        result = json.loads(subprocess.check_output([str(program)], text=True))
        result['source_archive_sha256'] = expected
        (folder/output).write_text(json.dumps(result, indent=2)+'\n')
        print('Regenerated pinned Stim oracle in', folder/output)
    # The surface oracle constructs geometry and signed flows in independent
    # Python integer algebra, then checks them against this freshly built CLI.
    cli = temp / 'stim'
    subprocess.run(['g++', '-std=c++20', '-O1', '-I'+str(root/'src'),
                    str(root/'src/main.cc'), *objects, '-pthread', '-o', str(cli)], check=True)
    subprocess.run([sys.executable, str(folder/'generate_surface_fixture.py'),
                    '--stim', str(cli), '--source-archive', str(archive),
                    '--output', str(folder/'surface-stim-1.15.0.json')], check=True)
