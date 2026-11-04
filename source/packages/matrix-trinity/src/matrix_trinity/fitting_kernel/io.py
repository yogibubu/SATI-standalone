from __future__ import annotations



def write_terms(path, coeff, terms):
    with open(path, "w") as fh:
        for c, term in zip(coeff, terms):
            idxs = []
            for i, exp in term.exps:
                idxs.extend([i + 1] * exp)
            fh.write(" ".join(str(i) for i in idxs) + f" {c:.16e}\n")
