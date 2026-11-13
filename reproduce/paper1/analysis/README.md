# Analysis scripts behind main-text Sections 4 and 5

The scripts in this folder produced the records in `../records/round3/`, from which the numbers of the main text (Section 4,
"Verification and validation"; Section 5, "Application"; Table 2; Figs 4 and 6) and of supplement S4 are taken. This file covers
the experiments on synthetic frames; the two comparisons with real data (`geocrack_comparison.py`, `fraxet_sites.py`) are
described in [`README_real_data.md`](README_real_data.md).

Recomputing the paper's numbers from the records needs none of these scripts. They are here so that every record can be
produced again from frames that the generator writes.

Terms are those of the main README. The manuscript calls the opening of a fracture its "aperture", and the trace segments and
windows that the labels mark as unclear (U) "unconfirmed"; the remaining visible length is "brightness-confirmed".

## Frames

Every script takes `--frames FRAMES`: a folder with one sub-folder per frame, named by the frame identifier and written by

```bash
python -m fracgen frame SCENE_DIR --out FRAMES --name FRAME_ID <frame_options>
```

`frames_io.py` describes the files of a frame folder that are read. The frames and the options that generate them are listed in

* `../frames.csv`: the 115 labelled frames of the paper (`a0001` ... `d0001`); column `block` gives the image set, `C` being the
  multi-illumination aerial set (50 frames, 5 viewpoints x 5 illumination conditions x 2 render seeds);
* `../sweeps.csv`: the frames of the view-angle sweep (`vs_*`) and of the illumination sweep (`ls_*`, `w2_*`).

A frame takes 5 to 25 minutes to generate (see the main README). The scenes are generated once and shared by their frames:
`python -m fracgen scene <scene_options> --out SCENE_DIR`, then `python -m fracgen cameras SCENE_DIR`.

Eleven viewpoints are used with one frame each (`frames_io.VIEWPOINTS`): the five viewpoints of the multi-illumination aerial
set (`a0077` flat outcrop; `a0074`, `a0062`, `a0099`, `a0044` standalone outcrops) and six further ones (`a0001`, `a0002`,
`a0018` flat outcrops; `a0061`, `d0001`, `a0020` standalone outcrops).

## Scripts

Commands are typed from the repository root. Every script writes its record to `reproduce/paper1/records/round3/` unless
`--out` names another file, and prints the main readings.

| Script | Paper item | Frames read | Record |
|---|---|---|---|
| `ridge_check.py` | Section 4.1, last paragraph (independent ridge filter, AUC); S4 ridge-filter table | the 50 frames of block C: image, region mask, label layers | `ridge_check.json` |
| `ablation.py` | Table 2, rows "Occlusion test", "Removed-volume check", "Position refinement"; Section 4.2 (unclear length per image); Section 5.3, second paragraph (length retained at the near-nadir viewpoint) | the 50 frames of block C: `report.json` | `ablation.json` |
| `u_threshold.py` | Table 2, row "Brightness check"; Section 4.2 (unclear windows at thresholds of 1, 2 and 4 levels; windows with insufficient sampling); Section 5.3 (windows at the near-nadir viewpoint); S4 threshold table | the 50 frames of block C: `windows.json` | `u_threshold.json` |
| `occluded_segments.py` | Section 4.2 (occluded traces on rock facing away from the camera); S4 "Position and facing direction of the occluded segments" with the distance table; S4, length retained at the viewpoint that meets the facing conditions | the 11 viewpoint frames, and the other 9 frames of every block-C viewpoint that meets the conditions (`a0063` ... `a0071`) | `occluded_segments.json` |
| `mask_skeleton.py` | Section 4.2, last paragraph (skeleton of the fracture mouth); S4 skeleton table, with the angle theta | the 11 viewpoint frames | `mask_skeleton.json` |
| `fragmentation.py` | S4 table of connected components of the final centreline | the 50 frames of block C: `report.json`, `target.npy`, `u_line.npy` | `fragmentation.json` |
| `width_curve.py` | Section 5.3, first paragraph (dark-band width against geometric opening); S4 "Apparent width ..." | 30 frames: 6 viewpoints x 5 mean openings (listed below); also reads `../records/fine_width_formal.json` | `width_curve.json` |
| `view_sweep.py` | Section 5.1 and Fig. 4 (view-angle sweep); S4 view-sweep table | `plan`: the two scenes; `collect`: the 17 frames `vs_*` | `view_sweep.json` |
| `light_sweep.py` | Section 5.2 (renders the sun directions of the illumination sweep) | the 6 base frames `ls_*`; writes the sweep images | images in `SWEEP/` |
| `light_sweep_stats.py` | Section 5.2 and Fig. 6 (illumination sweep, shadow ratio S); S4 illumination-sweep tables | the 6 base frames `ls_*`, the 6 whole-chain frames `w2_*`, the sweep images | `light_sweep.json`, `light_sweep_windows.npz` |

```bash
python reproduce/paper1/analysis/ridge_check.py --frames FRAMES
python reproduce/paper1/analysis/ablation.py --frames FRAMES
python reproduce/paper1/analysis/u_threshold.py --frames FRAMES
python reproduce/paper1/analysis/occluded_segments.py --frames FRAMES
python reproduce/paper1/analysis/mask_skeleton.py --frames FRAMES
python reproduce/paper1/analysis/fragmentation.py --frames FRAMES
python reproduce/paper1/analysis/width_curve.py --frames FRAMES
```

`occluded_segments.py`, `mask_skeleton.py` and `fragmentation.py` also take `--selftest`, which checks their functions on
synthetic shapes and reads no frame.

The 30 frames of `width_curve.py` are `a0001`-`a0012`, `a0017`, `a0018`, `a0020`, `a0022`-`a0026`, `a0041`, `a0054`, `a0061`,
`a0072`, `a0073`, `a0075`, `a0076` and `a0096`-`a0098`.

### View-angle sweep

```bash
python reproduce/paper1/analysis/view_sweep.py plan --scene flat=FLAT_SCENE_DIR --scene standalone=STANDALONE_SCENE_DIR --out view_sweep_plan.json
python reproduce/paper1/analysis/view_sweep.py collect --plan view_sweep_plan.json --frames FRAMES
```

`plan` computes the 18 camera positions from the camera plans of the two scenes (flat outcrop: seed 26120001; standalone
outcrop: seed 26130000; their scene options are those of frames `a0077` and `a0044` in `../frames.csv`) and writes, for every
viewpoint, the options of `fracgen frame` (the same options as the `frame_options` of `../sweeps.csv`) and geometric readings
of the camera position. The viewpoint at an off-nadir angle of 0 on the standalone outcrop lies inside the rock body; no frame exists
for it. `collect` reads the 17 generated frames and writes the record.

### Illumination sweep

```bash
python reproduce/paper1/analysis/light_sweep.py ls_v1b055 ls_v1w030 ls_v1w080 ls_v1d010 ls_v1d025 ls_v2b055 --frames FRAMES --sweep SWEEP
python reproduce/paper1/analysis/light_sweep_stats.py --frames FRAMES --sweep SWEEP
```

`light_sweep.py` renders, on the cut geometry of each base frame, 72 sun directions (and 6 repeats with another render seed for
the two baseline cases): 444 images, about 6 s each on the tested GPU. It uses the generator's render server, so the machine
configuration of the generator (interpreters, Blender) must be in place; `--dry-run` only compares the render specification it
builds with that of each base frame. `light_sweep_stats.py` recalculates the brightness difference of the windows of each base
frame on every image, bins the windows by shadow ratio, by the angle between sun azimuth and strike and by sun direction relative
to the fracture plane, and compares the procedure with the six frames `w2_*`, which were generated with the whole chain under
three of the sun directions. A sun direction that is missing in `SWEEP` is looked up as a whole-chain frame
`FRAMES/<case>_<aAAAeEEsS>` (the frame names of `../sweeps.csv`).

`light_sweep_windows.npz` holds, for every case, the brightness difference (adjacent-rock ring minus in-fracture band) of every
window under every sun direction as 32-bit floats, with the geometry of the windows (strike, fracture-plane normal, in-plane
direction, image length). The binned statistics of `light_sweep.json` can be recomputed from it with NumPy alone; the key
`window_table` of the record names the arrays and gives the formulas. It replaces the full table of the original analysis
(13.6 MB), of which it keeps the columns that the statistics of the paper use.

## Requirements

* The labels environment of the generator (NumPy, SciPy, Pillow). `ablation.py`, `u_threshold.py` and `view_sweep.py` need
  NumPy only.
* `light_sweep.py` and `light_sweep_stats.py` import the generator (`frgb`, `fracgen`) and must be run from the repository root.
* The analyses take seconds to a minute once the frames exist (`light_sweep_stats.py` about half a minute for the 444 images).
* No random numbers are used; a script run twice on the same frames writes the same record.

## What the records hold

Each record starts with `description`, `paper_items` and `units`, then the rows from which the paper's numbers are computed:
one row per image (`frames`), per viewpoint (`views`, `per_view`, `rows`) or per case (`points`, `cases`), followed by the
summaries that the manuscript quotes. Frame identifiers are those of `../frames.csv` and `../sweeps.csv`; `sid` and `pid` are
the scene and camera names of `../records/frames_index.json`.

The records were written by these scripts from the frames of the paper and of the two sweeps. The two baseline base frames
of the illumination sweep (`ls_v1b055`, `ls_v2b055`) were generated with `fracgen frame` and the options of the paper frames
`a0077` and `a0044`, and their label layers are bit-identical to those of the paper frames. Run on them, `ablation.py`,
`fragmentation.py`, `mask_skeleton.py` and `occluded_segments.py` gave the same rows as on the paper frames, and
`ridge_check.py` the same AUC values to four decimals (the two renders differ by one 8-bit level in a few tens of values).
