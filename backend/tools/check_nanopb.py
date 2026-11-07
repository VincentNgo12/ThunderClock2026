# File: check_nanopb.py
# Purpose: Build a host C harness to check Python/nanopb fixtures without touching ESP32 firmware.
# Contributors: Unassigned

import argparse
from pathlib import Path
import subprocess
import shutil
import time
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nanopb-root', type=Path, required=True)
    parser.add_argument('--cc', nargs='+', default=['cc'], help='C compiler, optionally followed by driver args such as zig cc')
    args = parser.parse_args()
    runtime = args.nanopb_root.resolve()
    if '#define NANOPB_VERSION "nanopb-0.4.9.1"' not in (runtime / 'pb.h').read_text():
        raise SystemExit('Use nanopb runtime 0.4.9.1 to match the generator')
    directory = tempfile.mkdtemp()
    try:
        executable = Path(directory) / ('conformance.exe' if sys.platform == 'win32' else 'conformance')
        subprocess.run([*args.cc, '-std=c11', '-Wall', '-Wextra', '-Werror', f'-I{runtime}', f'-I{ROOT / "protocol/generated"}',
                        str(ROOT / 'protocol/fixtures/nanopb_fixture.c'),
                        *(str(ROOT / f'protocol/generated/{name}.pb.c') for name in ('orientation', 'telemetry')),
                        *(str(runtime / name) for name in ('pb_common.c', 'pb_encode.c', 'pb_decode.c')),
                        '-o', str(executable)], check=True)
        fixtures = ROOT / 'protocol/fixtures'
        expected = (fixtures / 'quaternion_identity.bin').read_bytes()
        if subprocess.check_output([str(executable), 'emit']) != expected:
            raise SystemExit('Independent C identity encoding differs from Python')
        for name in ('raw_signed', 'quaternion_identity', 'quaternion_partial', 'quaternion_maximum'):
            data = (fixtures / f'{name}.bin').read_bytes()
            result = subprocess.check_output([str(executable), 'raw' if name.startswith('raw') else 'quaternion'], input=data)
            if result != data:
                raise SystemExit(f'nanopb/Python mismatch: {name}')
            print(f'PASS nanopb encode/decode: {name}, {len(data)} bytes')

    finally:
        # Windows may briefly retain an executable mapping after process exit.
        for attempt in range(6):
            try:
                shutil.rmtree(directory)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.5)


if __name__ == '__main__':
    main()
