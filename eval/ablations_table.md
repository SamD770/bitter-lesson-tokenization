| Ablation | Downsample Rate at convergence | FineWeb Validation BPB | Fraction of Whitespace with token boundaries |
|---|---|---|---|
| Baseline | 0.196 | 1.548 | 0.99 |
| no time discounting ($\gamma=1$) | 0.215 | 1.542 | 0.41 |
| no batch-relative advantages | 0.208 | 1.543 | 0.99 |
| no early-exit relative rewards | 0.200 | 1.562 | 0.72 |
| no sliding window ($w=1$) | 0.196 | 1.51 | 0.99 |
| no consistency loss ($\lambda_{target} = 0$) | 1.0 | 1.39 | 1.0 | 