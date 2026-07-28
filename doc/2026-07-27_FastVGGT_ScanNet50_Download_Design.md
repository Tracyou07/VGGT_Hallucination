# FastVGGT ScanNet-50 Download Design

## Goal

Prepare the exact 50 ScanNet v2 scans listed by FastVGGT for the Round 2
local-global study and later dense-geometry evaluation. The workflow must use
the official ScanNet distribution, require prior Terms of Use acceptance, and
never stop for an interactive key press.

The scene IDs are copied verbatim from FastVGGT's
`eval/scannet_50.yaml`:

`https://github.com/mystorm16/FastVGGT/blob/main/eval/scannet_50.yaml`

## Artifacts

Add `configs/fastvggt_scannet50.txt` with exactly 50 unique scene IDs. For each
scene, prepare:

- `<scene>.sens` for RGB frames, raw camera poses, and intrinsics;
- `<scene>_vh_clean_2.ply` for later raw-GT point-cloud evaluation.

Use these roots by default:

```text
/root/autodl-tmp/datasets/scannetv2/raw_sens/scans/<scene>/<scene>.sens
/root/autodl-tmp/datasets/scannetv2/raw/scans/<scene>/<scene>_vh_clean_2.ply
/root/autodl-tmp/datasets/scannetv2/process_scannet/<scene>/
```

The processed directory contains extracted `color/` and raw `pose/` data
expected by the existing VGGT runners. Intrinsics remain available in the
original `.sens` file.

## Unified Download Workflow

Extend the existing `scripts/autodl/prepare_scannet_camera_iteration.sh`
instead of adding a second preparation script. The current 10-scene camera
iteration workflow and FastVGGT ScanNet-50 then share the same official
downloader, retry behavior, directory layout, and `.sens` extractor.

The script:

1. exits unless `SCANNET_TOS_ACCEPTED=1`;
2. reads any configured `SCENE_LIST` and applies `SCENE_LIMIT`, where `0`
   selects the whole list;
3. validates unique, well-formed scene IDs;
4. downloads or reuses each `.sens` and, when requested,
   `_vh_clean_2.ply`;
5. pipes confirmation input to the official downloader, so no manual space or
   key press is required;
6. invokes each attempt with a fresh per-asset staging root beside the final
   destination, isolating all downloader-created files on the same filesystem;
7. validates the exact staged nonempty file, moves it to the requested
   destination only after downloader success, and removes the entire staging
   root after every attempt;
8. removes stale staging roots for the same asset on restart, while reusing a
   valid nonempty destination;
9. extracts `.sens` files and verifies every selected processed scene.

An interrupted run is restarted with the same command. Valid final files are
reused; zero-byte destinations are removed before the first real attempt, and
randomly named downloader partials remain confined to staging roots. `RAW_DIR`
is the exact `.sens` destination and extractor input. `RAW_DOWNLOAD_ROOT`
defines only the default `RAW_DIR=$RAW_DOWNLOAD_ROOT/scans`, so an explicit
`RAW_DIR` wins even when both variables are set. PLY files remain under
`GT_DOWNLOAD_ROOT/scans/<scene>/`. The existing 10-scene command keeps its
current behavior because PLY download remains opt-in.

## Controls And Testing

Environment overrides include `AUTODL_TMP`, `SCANNET_ROOT`, `SCENE_LIST`,
`SCENE_LIMIT`, `DOWNLOAD_RETRIES`, `DOWNLOAD_GT_PLY`, and the existing conda
variables. `DOWNLOAD_GT_PLY` defaults to `0` for backward compatibility; the
ScanNet-50 command explicitly sets it to `1`.

CPU-only tests lock the FastVGGT configuration to exactly 50 unique expected
scene IDs. Behavior tests run the main script against a fake official
downloader to cover zero-byte replacement, failed random-partial cleanup,
valid-file reuse, and arbitrary `RAW_DIR` overrides. Static contract checks
retain the ToS guard, defaults, extraction, and absence of environment/weight
installation.

Run the prepared workflow with:

```bash
SCANNET_TOS_ACCEPTED=1 \
SCENE_LIST=configs/fastvggt_scannet50.txt \
SCENE_LIMIT=0 \
DOWNLOAD_GT_PLY=1 \
bash scripts/autodl/prepare_scannet_camera_iteration.sh
```
