# Stage A decision: move the flow state to camera translation

The formal Stage A run
`privileged_teacher_lift_20260829T012716Z_tolfix` completed cleanly at commit
`cee41a09ac4085c8d6b0b343ca07d8e8c53ace3c`.  Its launcher returned `0`, all stderr files
were empty, and an independent verifier reproduced the signed inventory.  The scientific
classification was nevertheless `LATENT_LIFT_FAILED`.

Aggregate token-lift result:

- mean full-scene translation utility: `+7.0175299981%`;
- positive scenes: `9/10`;
- mean teacher retention: `0.6968048295`;
- minimum scene utility: `-2.5722388277%` on `scene0029_01`;
- mean rotation delta: `+2.2290900231 deg`;
- maximum uncovered drift: `0.1202015950` on `scene0000_00`.

The failure is not corruption and does not refute the short-window signal.  Short teachers
were selected only by translation, but their covered-frame rotation is on average
`+2.2970857413 deg` worse than the long baseline.  The token lift reproduces that bad label.
Its absolute-trajectory smoothness term also has nonzero gradient at the exact no-op and can
move uncovered frames.

A CPU-only counterfactual retained the long rotation/FOV, copied short-teacher centers only on
covered frames, and kept the baseline elsewhere.  It produced:

- mean full-scene utility: `+10.6546%`;
- positive scenes: `10/10`;
- worst scene: `+0.8438%`;
- teacher retention: `1.0`;
- rotation delta: `0`;
- uncovered drift: `0`.

The next source-level experiment therefore uses normalized dense corrections to Camera Head
`w2c` translation rather than high-dimensional Camera-token DCT residuals.  Long and short
Camera tokens remain the posterior conditions, and the deployment prior remains long-only.
The formal one-500-frame-backbone contract is unchanged.
