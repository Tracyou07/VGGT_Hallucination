# VGGT

This branch contains a minimal VGGT inference baseline. Research experiments,
generated results, and dataset-specific automation are developed on dedicated
Git worktree branches so the baseline remains stable.

## Setup

VGGT requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
pip install -e .
```

Verify the package import after installation:

```bash
python -c "from vggt.models.vggt import VGGT; print(VGGT.__name__)"
```

## Source Layout

- `vggt/models/` contains the top-level model and token aggregator.
- `vggt/heads/` contains camera, depth, point, and tracking heads.
- `vggt/layers/` contains Transformer and patch-embedding components.
- `vggt/utils/` contains image loading, geometry, pose, and visualization helpers.

The research worktrees are local development environments and are intentionally
excluded from this baseline branch.
