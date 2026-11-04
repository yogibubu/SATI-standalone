# ORACLE publication examples

Run every example from any working directory with:

```bash
oracle analyze benzene.xyz -o benzene.xyzin \
  --report benzene.oracle.json --snapshot benzene.topology.json
```

The set exercises aromatic, fused-ring, strained cage, transition-metal,
quasi-symmetric and noncovalent structures. `quasi_symmetric_water.xyz` is
intentionally slightly displaced and demonstrates threshold-controlled
symmetrization. `water_dimer.xyz` contains two covalent fragments joined by a
calibrated O--H···O(H) pseudo-bond; it is the minimal L1-to-PL1 example.
`saccharin.xyz` freezes a planar fused-ring heterocycle containing C, H, N, O
and S. It is a publication regression for full-rank redundant PIC/B
construction after the ring out-of-plane convention was introduced.
