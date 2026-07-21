# Camera Head Amplification Experiment Design

## Goal

Round 1.6 determines whether VGGT's four-block Camera Head amplifies the
normalized Camera Token drift observed in Round 1.5. It is a frozen-weight,
training-free replay experiment and does not rerun the image encoder or
Aggregator.

## Controlled Replays

For every scene, use the 200-frame selection and the same frame IDs inside the
500-frame selection:

- `H200(Z200)`: baseline 200-frame Camera Tokens decoded at length 200;
- `H200(Z500_shared)`: the corresponding 500-context tokens decoded at length
  200, isolating input-token drift at fixed Camera Head sequence length;
- `H500(Z500)`: all 500-context tokens decoded at length 500, validating replay
  against the original Round 1.5 prediction.

The shared outputs of the second and third replay are also compared. Their
shared-frame input token values are identical, so their layer drift measures
the effect of adding 300 Camera Head context tokens and has no amplification
ratio denominator.

The first and third replays must reproduce the saved raw predictions within a
configured tolerance before amplification conclusions are emitted.

## Instrumentation And Metrics

Forward hooks observe each of the four trunk blocks without changing the
production Camera Head API. The replay records compact per-frame norms after
AdaLN input modulation, each trunk block, trunk normalization, the 9D delta,
and accumulated raw pose for all four refinement iterations. High-dimensional
features remain transient.

For matched replays, report RMS representation drift and amplification ratio
relative to normalized input-token RMS drift. Also report raw 9D drift and
independently aligned pose ATE/ARE against raw GT. A ratio above one is a
diagnostic gain, not by itself causal proof.

## Reproduction And Outputs

The CLI consumes a published Round 1.5 run plus the local official VGGT
checkpoint. An AutoDL wrapper fixes contexts to 200 and 500 and iterations to
four. Outputs are CSV/JSON numeric summaries under an external run
directory. Only compact summaries and per-frame scalar arrays may be exported
to `results/camera_head_amplification/<run_id>`; checkpoints, datasets, and
2048-dimensional activations are never committed.

## Validation

CPU tests cover frame matching, replay capture ordering, amplification math,
baseline validation, CLI contracts, and exporter whitelists using a small
Camera Head. AutoDL performs the real checkpoint replay on GPU; CPU remains a
supported debug device.
