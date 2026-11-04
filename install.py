"""Install the supplied, hash-verified wheels without contacting an index."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
record = json.loads((root / 'wheelhouse-manifest.json').read_text())
for name, expected in record['files'].items():
    with (root / 'wheels' / name).open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected:
        raise ValueError(f'Wheel hash mismatch: {name}')
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--find-links',
                str(root / 'wheels'), 'sati-standalone==0.1.0rc8', 'pytest>=8,<9'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
