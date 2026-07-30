# Hidden Interpolation Results

## Intervention

The interpolation is applied only to the 41 selected hidden units in Camera
Head refinement iteration 0:

```text
h_new = h_long + alpha * (h_short - h_long)
```

It is not a fusion or replacement of the complete short-context
representation. All other iteration-0 units and all units in later refinement
iterations remain unchanged.

## Calibration Alpha Sweep

| Alpha | Selected delta estimate | 95% CI | Improved scenes | Control delta | Interpretation |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.01 | `-0.0001403767` | `[-0.0003082785, +0.0000234858]` | `0.6` | `+0.0000015981` | Candidate local effective band |
| 0.02 | `-0.0001866527` | `[-0.0005402582, +0.0001167955]` | `0.4` | `+0.0000039288` | Best calibration estimate; selected for holdout |
| 0.05 | `-0.0001162461` | `[-0.0007747349, +0.0004650559]` | `0.4` | `+0.0000134060` | Candidate local effective band; upper boundary |
| 0.10 | `+0.0004676500` | `[-0.0020594472, +0.0030438683]` | `0.4` | `+0.0000346352` | Effect has turned unfavorable |
| 0.25 | `+0.0095738969` | `[-0.0049301678, +0.0246664207]` | `0.3` | `+0.0000861619` | Outside the useful local regime |
| 0.50 | `+0.0730599422` | `[+0.0236114459, +0.1238236631]` | `0.2` | `+0.0001171555` | Clearly harmful |
| 1.00 | `+0.2797968102` | `[+0.1560268186, +0.4134725973]` | `0.1` | `+0.0003096158` | Full replacement; strongly harmful |

The calibration trend places the candidate locally effective band at
`0.01-0.05`, with a sign/regime transition between `0.05` and `0.10`.
All three candidate-band confidence intervals cross zero on the 10-scene
calibration split, so the sweep is used to select a hypothesis rather than to
claim calibration-set significance.

## 40-Scene Holdout

Only the calibration-frozen `alpha=0.02` was independently evaluated on the
40-scene holdout.

- Mean aligned translation error: `0.0880634 -> 0.0878139`
- Translation delta: `-0.000249461`
- Translation-delta 95% CI: `[-0.000424608, -0.000080119]`
- Improved scenes: `26/40` (`65%`)
- Mean aligned rotation delta: `-0.003870 deg`
- Mean camera-center displacement: `0.001287`

The selected intervention beat the evaluated random control in `26/40`
scenes. Its mean translation delta relative to that control was
`-0.000263442`, with 95% CI
`[-0.000439055, -0.000097912]`.

## Reporting Limitation

The frozen manifest defines five random controls, but this run evaluated only
`control_00`. Consequently, the selected-versus-baseline holdout result is a
40-scene result, while the selected-versus-control conclusion is based on one
random control set rather than the configured five-control average. The
remaining controls must be evaluated before making the formal multi-control
claim.
