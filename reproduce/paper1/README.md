# Reproducing the paper's results

The paper's frames were produced with the code in this repository (the internal version differed only in comments,
documentation strings and the declared edits listed in `docs/provenance.md`). This folder holds what is needed to
reproduce its numbers at three levels of effort.

| Level | What | Time |
|---|---|---|
| 1 | [numbers, tables and figures from the stored records](#1-numbers-tables-and-figures-from-the-records-seconds) | seconds |
| 2 | [frames from their parameters](#2-frames-of-the-paper-from-their-parameters-minutes-per-frame), and [the records from the frames](#from-frames-to-records) | minutes per frame |
| 3 | [the end-to-end example](#3-the-end-to-end-example-about-16-minutes) | about 16 minutes |

```
main_text.py, main_text_values.json   numbers of the main text, recomputed from records/ and compared with the printed values
tables.py, tables.json                tables of the supplement (S1-S3), recomputed from records/
records/                              per-frame and per-case records of the labelled frames (supplement S1-S3)
records/round3/                       records behind main-text Sections 4 and 5 and supplement S4
figures/                              scripts that draw Figs 4, 5 and 6 from records/round3/
frames.csv, frames.json               the 115 labelled frames of the paper and the options that generate them
sweeps.csv, sweeps.json               the 473 frames of the view-angle sweep and the illumination sweep (Section 5)
analysis/                             scripts that produced records/round3/ from frames and from the public real datasets
```

Commands are typed from the repository root, as in the main README. Terms are those of the main README; the manuscript
writes "aperture" for the opening of a fracture and "unconfirmed" for the unclear (U) layer of the labels.

## 1. Numbers, tables and figures from the records (seconds)

### Main text

```bash
python reproduce/paper1/main_text.py
```

`main_text_values.json` lists the statements of the manuscript that carry numbers: the abstract, Sections 4 ("Verification
and validation"), 5 ("Application"), 6 and 7, Table 2 and the captions of Figs 4 to 6. Each item gives the place in the paper,
the values as printed, their rounding, and the record and keys behind them. `main_text.py` computes every item again from
the records, rounds it as printed and compares; it prints one line per item,

```
ok   view_flat                    rows     printed 10.0 27.5 46.9 8.3      recomputed 10.0 27.5 46.9 8.3      Section 5.1, paragraph 1; Fig. 4; ...
```

and ends with `all main-text values recomputed` (62 items with 215 printed values), or lists the items that differ and
exits with status 1. It needs the standard library and NumPy only.

How far down a recomputation starts is given by the `basis` of the item:

| Basis | Items | Meaning |
|---|---|---|
| `rows` | 34 | aggregated from the rows of a record: one row per image, per viewpoint or per case |
| `windows` | 5 | recomputed from the window table of the illumination sweep (`records/round3/light_sweep_windows.npz`): the brightness difference of every window under every sun direction, binned by sun elevation, shadow ratio S and sun side |
| `counts` | 11 | computed from stored counts: sample points of the occlusion analysis on the real outcrops, histograms of the measurement points (overlapping coefficients) |
| `stored` | 12 | a statistic read from the record as stored; computing it again needs the frames or the public dataset and the analysis script |

The `stored` items are the quantiles over the measurement points of the real images (medians of relative contrast, relative
width, centre brightness and dark-band width of GeoCrack and FraXet, the shares of the FraXet points inside the synthetic
range, the Spearman correlation with the illumination azimuth) and the corresponding quantiles of the synthetic images in
the two comparisons, the overlapping coefficient of the relative contrast at the FraXet sites, the pixel sizes of the sites
taken from the literature and the range of pixel sizes of the synthetic aerial images, the grid spacing of a DEM, the dip of
the fractures and the floor visibility at the flat-outcrop viewpoint of the illumination sweep, and the largest centre
deviation of the labels.

Records behind the main text:

| Record (`records/round3/`) | In the paper | Written by (`analysis/`) |
|---|---|---|
| `ridge_check.json` | Section 4.1, ridge filter (AUC) | `ridge_check.py` |
| `ablation.json` | Table 2, rows "Occlusion test", "Removed-volume check", "Position refinement"; Sections 4.2 and 5.3, unclear (unconfirmed) and retained length | `ablation.py` |
| `u_threshold.json` | Table 2, row "Brightness check"; Sections 4.2 and 5.3, unclear windows and their two causes | `u_threshold.py` |
| `occluded_segments.json` | Section 4.2, occluded traces on rock facing away from the camera | `occluded_segments.py` |
| `mask_skeleton.json` | Section 4.2, skeleton of the fracture mouth against the visible centreline | `mask_skeleton.py` |
| `geocrack_comparison.json` | Section 4.3, comparison with the GeoCrack photographs | `geocrack_comparison.py` |
| `fraxet_sites.json` | Section 4.3 (dark-band width at the FraXet sites), Section 5.1 and Fig. 5 (occlusion on real-outcrop DEMs), Section 5.2 (illumination azimuth) | `fraxet_sites.py` |
| `view_sweep.json` | Section 5.1 and Fig. 4, view-angle sweep | `view_sweep.py` |
| `light_sweep.json`, `light_sweep_windows.npz` | Section 5.2 and Fig. 6, illumination sweep | `light_sweep.py`, `light_sweep_stats.py` |
| `width_curve.json` | Section 5.3, dark-band width against geometric opening | `width_curve.py` |
| `fragmentation.json` | supplement S4 only (connected components of the final centreline) | `fragmentation.py` |

The records also hold the rows and summaries behind the tables of supplement S4; every record starts with a description of
its content and the paper items it serves. Four records of `records/` outside `round3/` are read as well: `m_summary_formal.json` and
`values_relief_formal.json` (the 54 aerial images of Section 4.1, the range of pixel sizes), `b_summary_formal.json`
(the shape-parameter set and the locality of its variants) and `m2_summary_formal.json` (Table 2, row "Mid-surface
positioning"); these records name standalone outcrops "isolated".

Not recomputed, because no record of the package covers them: Table 2, rows "Per-zone level selection" (shape error one
subdivision level lower) and "Relief applied along the normal" (deviation of the relief amplitude), which each come from one
case. Not compared either: the settings quoted in the text (limits of the acceptance criteria, width of the trace
neighbourhood, distance classes of the ridge-filter check, bin edges and thresholds, size of the detail of Fig. 3), round
figures that restate a compared range ("about 30 mm/px" for site EN, whose pixel sizes of 29 and 32 mm/px are compared) and
counts that describe the design (cases, fractures in frame).

### Tables of the supplement

```bash
python reproduce/paper1/tables.py
```

`records/` holds, for every labelled frame or shape-parameter case of the paper, the per-frame statistics that the
tables of the supplement (S1-S3) summarise. `tables.py` aggregates them again and compares each result with the summary
stored next to the rows:

| Paper table | Record | Recomputed |
|---|---|---|
| Per-frame statistics of the multi-light aerial set (50 frames) | `stats_e_formal.json` (`main_rows`) | median and range of every statistic, geometric opening per view |
| Locality and label checks of the shape-parameter variants | `b_summary_formal.json` (`rows`) | passes per parameter group, range of the in-band image change T |
| Dark-band width per view and mean opening | `fine_width_formal.json` (`rows`) | median width per view and opening |
| Formal label set | `frames_index.json` | frames per experiment block |

The script ends with `all recomputed summaries equal the stored ones` and writes `tables.json`. Other records used in the
text: `m_summary_formal.json` (occlusion, below-ground and position statistics), `m2_summary_formal.json` (source surface
against the cavity mid-surface), `m5_formal.json` (trace distances between each variant and its baseline), `k_exposure_formal.json` and
`exposure_reeval_formal.json` (exposure limits), `k_noise.json` (render-noise ruler), `f_ab_formal.json` (wall-panel
pairs), `b_cases.json` (the shape-parameter cases with their camera, parameters and reserved memory), `cost_relief_formal.json` (faces, memory, time and disk per frame), `values_relief_formal.json` (collected values).
Paths inside the records are relative names of folders and files of the original runs. They identify frames, cases and the
files a value was taken from; these files are not part of the package, and the scripts open none of them (two analysis
scripts take the frame identifier from the last part of `label_dir` in `fine_width_formal.json`). Keys with `box` in their
name hold values, or names of files and folders, of an earlier series of runs made with a flat cavity floor, before the floor
relief of the released version; they are kept for comparison. Keys with `yi` hold the readings of an earlier label run of the
same frames (frame identifier, reserved and peak memory, time, exit code; `points_not_in_yi_not_added` of `frames_index.json`
is an empty list), and `decl` is the memory reserved for a job. The time stamps (`at`, `finished` and `built` at the top of a
record, `indexed` of the frames in `frames_index.json`, where frame `d0001` has none) and the
SHA-256 values of files that are not in the package (`criterion_sha256`, `criterion.sha256`, `module_sha256`,
`analyze_opening_sha256`, `from_index.sha256`) are kept as provenance of the records; `png_sha256` and `target_sha256` in
`fine_width_formal.json` are the hashes of the image and of the target layer on which a width was measured, and
`analysis/width_curve.py` compares them with the frames it is given.

Blocks (`block` in `frames.csv` and `frames_index.json`): `B`, the shape-parameter cases (baselines and variants) built in the
runs behind these records; `R`, shape-parameter cases taken over from earlier runs with the same version of the cavity (their
cut mesh and the render of their `sun0` frame were not made again, and the four overcast frames of the block were rendered on
these cut meshes; `cost_relief_formal.json` lists the 16 cases under `B` with `reused` true); `C`, the multi-illumination
aerial set; `D`, the standalone-outcrop viewpoint at 10.70 mm/px; `F`, the close-range viewpoints at four scales. The groups
C, D, E, F and B of `m_summary_formal.json` and `m2_summary_formal.json` are defined in the `rule` text of
`m_summary_formal.json`. The multi-light aerial set of these records is the multi-illumination aerial set of the manuscript
and of the records in `round3/` (block `C` of `frames.csv`).

### Figures 4 to 6

```bash
python reproduce/paper1/figures/fig4_view_sweep.py --out my_figures
python reproduce/paper1/figures/fig5_real_occlusion.py --out my_figures
python reproduce/paper1/figures/fig6_illumination_sweep.py --out my_figures
```

Each script reads its records from `records/round3/` (`view_sweep.json`; `fraxet_sites.json` and `view_sweep.json`;
`light_sweep.json`), writes a PDF and a PNG and prints the plotted values. Without `--out` the files are written in the
current folder. The scripts need Matplotlib, which the two environment files of the generator do not install
(`pip install matplotlib`; 3.10.8 was used). On the test machine the three PNG files are byte-identical to the figures of
the manuscript.

## 2. Frames of the paper from their parameters (minutes per frame)

`frames.csv` and `frames.json` list the 115 frames of the paper's formal label set with their parameters: scene
configuration, scene seed, relief amplitude a and slope H, number of random faults, fracture parameters, sun direction or
overcast sky, resolution, samples, render seed and the camera shot (position, target, focal length). Each row also holds
the options that generate the frame again:

* `scene_options`: the arguments of `python -m fracgen scene` (add `--out SCENE_DIR`);
* `frame_options`: the arguments of `python -m fracgen frame SCENE_DIR` (add `--out FRAME_DIR` and `--device OPTIX`).

Run `python -m fracgen cameras SCENE_DIR` between the two. Frames with a plan camera use `--camera` with the row's
`camera_slot`; the close-range and near views (36 frames, `camera_slot` empty) give their camera with `--position`,
`--target` and `--focal-mm`, which replace the plan camera of slot 0. Overcast frames use `--sky overcast`, and
render-noise repeats use `--render-seed 1`.

Frame `a0044`, ready to copy: one command per line, typed from the repository root as in the main README; rendering uses
the GPU. It is the scene, camera and light of the end-to-end example, rendered as in the paper at 1024 px with 1024
samples.

```bash
python -m fracgen scene configs/scenes/standalone_outcrop.json --seed 26130000 --amplitude 2.0667663012280917 --hurst 0.9566568215681955 --outcrop standalone --faults 0 --out C:/fg/s26130000
python -m fracgen cameras C:/fg/s26130000
python -m fracgen frame C:/fg/s26130000 --out C:/fg/a0044 --camera 1 --width-m 0.055 --variation 0.225 --depth-m 0.17500000000000002 --roughness-m 0.0020625 --sun-azimuth 102.16591636974047 --sun-elevation 17.5 --resolution 1024 --spp 1024 --device OPTIX
```

If the end-to-end example has already been run with `--out C:/fg/example`, its scene can be used instead and the first
two lines skipped: give `C:/fg/example/scene` in place of `C:/fg/s26130000` in the last line. The camera plan is drawn
from the scene seed, so `camera_slot` gives the paper's camera; for scene 26130000 the whole plan was checked to be
identical.

On the test machine the last command took 9 minutes. Compared with the paper's frame a0044, every label layer was
bit-identical (893 positive pixels, visible trace length 1155.718 px), and so were the depth, material index and region
mask of the render; the new GPU render differed from the paper's by at most 1e-5 in linear RGB and by one 8-bit level
in 0.0007 % of the PNG values.

A close-range frame with its own camera was checked in the same way: frame `a0040` (1.0 mm/px), rendered on the same
scene with the row's `frame_options` (`--camera 0 --position ... --target ... --focal-mm 35.0`), took 5 minutes. Every
label layer was bit-identical (1004 positive pixels, visible trace length 1039.096 px), and so were the depth, material
index and region mask; the render differed from the paper's by at most 2.5e-5 in linear RGB and by one 8-bit level in
0.0016 % of the PNG values.

`--sky overcast` and `--render-seed 1` pass the paper's settings to the same frame driver, but no frame that needs them was
regenerated for comparison.

Rendering on the GPU is repeatable to within a small noise (in the paper, repeated GPU renders of the same frame differed
by at most 6e-5 in linear RGB and by one 8-bit level in 0.0036 % of the PNG values); depth, material index and mask are
identical. The label geometry does not depend on the image; only the unclear (U) test reads the RGB values.

### The two sweeps of Section 5

`sweeps.csv` and `sweeps.json` list the 473 frames of the two controlled experiments, with the columns of `frames.csv`, five
more (`role`, `viewpoint`, `beta_deg`, `camera_azimuth_deg`, `azimuth_kind`) and the same two option strings. Both use scenes
of the formal label set (seed 26120001, the flat outcrop of frame `a0077`, and seed 26130000, the standalone outcrop of
frame `a0044` and of the end-to-end example).

* **View-angle sweep** (block `view_sweep`, 17 frames `vs_*`): one target point per outcrop, constant distance, off-nadir
  angles of 0, 15, 30, 45 and 60 degrees with the camera azimuth across or along the strike of the target fracture, each
  given as a user-defined camera (`--position`, `--target`, `--focal-mm`). The nadir view of the standalone outcrop is not
  among them: its camera position lies inside the rock body (`not_generated` in `sweeps.json` gives its options).
* **Illumination sweep** (block `light_sweep`, 456 frames): 6 cases (five combinations of mean opening and depth at the
  flat-outcrop viewpoint V1, the baseline at the standalone-outcrop viewpoint V2). Each case has one `base` frame, 72
  `sweep` rows (sun azimuth in steps of 15 degrees at elevations of 20, 40 and 60 degrees); the two baseline cases have 6
  `repeat` rows with render seed 1 and 3 `full_chain` frames each.

The `frame_options` of every row generate the frame with the whole chain. In the runs behind the paper only the base and
`full_chain` frames of the illumination sweep were generated that way: the sweep and repeat images were rendered on the cut
geometry of the base frame of their case (about 6 s per image instead of 10 to 17 minutes), which
`analysis/light_sweep.py` does from a base frame. The two routes were compared on the 6 `full_chain` frames (identical
windows, depth, material index and mask; every window classified identically); `sweeps.json` gives the details.
The frames of the two sweeps were generated with this package (version 1.0.0, before the last rewording of comments, messages
and record strings; see `docs/provenance.md`) and were not generated a second time for
comparison; the option strings of `sweeps.csv` were checked against the records of those runs and parsed with the argument
parsers of `fracgen scene` and `fracgen frame`.

### From frames to records

The scripts in `analysis/` produced the records of `records/round3/` and write them again from frames generated as above
(`--out FRAMES --name FRAME_ID`, so that every frame lies in `FRAMES/FRAME_ID`) and, for the two comparisons with real
data, from the public datasets:

```bash
python reproduce/paper1/analysis/ablation.py --frames FRAMES
python reproduce/paper1/analysis/geocrack_comparison.py --geocrack GEOCRACK --reference reproduce/paper1/records/round3/geocrack_comparison.json
```

[`analysis/README.md`](analysis/README.md) lists every script with the frames it reads, the record it writes and its
requirements (the labels environment; `fraxet_sites.py` also needs PyTorch).
[`analysis/README_real_data.md`](analysis/README_real_data.md) describes the two datasets, GeoCrack (Harvard Dataverse,
<https://doi.org/10.7910/DVN/E4OXHQ>, CC0 1.0) and FraXet (Zenodo, <https://doi.org/10.5281/zenodo.17069947>, CC BY 4.0),
with their references, how to obtain them and which files are used. No file of either dataset is redistributed here; the
records hold only statistics derived from them.

What was run again with the released scripts: the scripts of the synthetic experiments wrote, from the paper's frames and
the frames of the two sweeps, records equal to those behind the paper; `geocrack_comparison.py` was run on the synthetic
frames and on the 49 photographs, and `fraxet_sites.py` on the three sites, and both gave the stored records (two rank
correlations of `fraxet_sites.py` differ in the fifth and sixth decimal). The two README files of `analysis/` give the details.

## 3. The end-to-end example (about 16 minutes)

`python -m fracgen example` regenerates scene 26130000 of the paper from its seed and runs one small frame through the
whole chain; see the main README.
