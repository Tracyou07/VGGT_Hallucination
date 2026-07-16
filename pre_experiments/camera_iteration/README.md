# Camera Iteration Study

This package will expose and evaluate Camera Head iterations `1, 2, 4, 8, 16`
without training or parameter updates. The implementation follows:

- `doc/2026-07-16_Camera_Iteration_Worktree_Design.md`
- `doc/VGGT_DiT_Research_Guide.md`
- `doc/VGGT_DiT_Implementation_Plan.md`

The AutoDL runner assumes existing weights and ScanNet data. Final commands,
output schemas, resume behavior, and metric definitions are added with the
implementation; no experiment has run yet.
