from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .vibrational_internal import modes_from_gaussian_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Gaussian .log/.out file")
    ap.add_argument("--fchk", required=False, help="Gaussian .fchk/.fch file")
    ap.add_argument("--out", required=True, help="Output prefix")
    ap.add_argument("--scale-json", required=False, help="JSON dict of Pulay scale factors")
    args = ap.parse_args()

    scale_map = None
    if args.scale_json:
        scale_map = json.loads(Path(args.scale_json).read_text())

    freqs, modes_q, U, prims = modes_from_gaussian_log(
        Path(args.log), fchk_path=Path(args.fchk) if args.fchk else None, scale_map=scale_map
    )

    out = Path(args.out)
    np.save(out.with_suffix(".freqs.npy"), freqs)
    np.save(out.with_suffix(".modes_q.npy"), modes_q)
    np.save(out.with_suffix(".U.npy"), U)
    out.with_suffix(".freqs.txt").write_text("\n".join(f"{f: .6f}" for f in freqs) + "\n")


if __name__ == "__main__":
    main()
