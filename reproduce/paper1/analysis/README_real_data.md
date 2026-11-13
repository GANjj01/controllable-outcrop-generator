# Real data behind Sections 4.3, 5.1 and 5.2

Two public datasets are compared with the generator's frames in the paper. **No file of either dataset is redistributed
in this repository**: the two records hold only statistics derived from them, and the scripts read the data from a folder
given on the command line.

| Record (`../records/round3/`) | Script (this folder) | Data | In the paper |
|---|---|---|---|
| `geocrack_comparison.json` | `geocrack_comparison.py` | GeoCrack, 49 photographs with edge masks | Section 4.3, first paragraph; supplement S4, "Comparison with real photographs" |
| `fraxet_sites.json` | `fraxet_sites.py` | FraXet, sites OG1, KB3 and EN | Section 4.3, second paragraph; Section 5.1, last paragraph and Fig. 5; Section 5.2, last paragraph; supplement S4, "Comparison on real-outcrop data" |

## GeoCrack

* **Dataset:** GeoCrack, version 1.0, Harvard Dataverse, <https://doi.org/10.7910/DVN/E4OXHQ>.
* **Licence:** CC0 1.0.
* **References:**
  * Ansari, Mohammed Yaqoob, 2024. GeoCrack: A high-resolution dataset of fracture edges in geological outcrops
    (version 1.0) [data set]. Harvard Dataverse. <https://doi.org/10.7910/DVN/E4OXHQ>.
  * Yaqoob, M., Ishaq, M., Ansari, M.Y., Konagandla, V.R.S., Al Tamimi, T., Tavani, S., Corradetti, A., Seers, T.D., 2024.
    GeoCrack: A high-resolution dataset for segmentation of fracture edges in geological outcrops. Scientific Data 11,
    1318. <https://doi.org/10.1038/s41597-024-04107-0>.
* **How to obtain it:** download the files of the two folders `Original Images` and `Edge Masks` from the Dataverse page
  into one folder, keeping the folder names. A photograph and its mask are PNG files of the same name (for example
  `Original Images/DJI_0011.png` and `Edge Masks/DJI_0011.png`).
* **What is used:** the 49 photographs listed in `IMAGES` of `geocrack_comparison.py` (and, with their statistics, under
  `per_image.real` of the record), each with its edge mask of the same size. The photographs have up to 50 megapixels.

```bash
python reproduce/paper1/analysis/geocrack_comparison.py --geocrack GEOCRACK --reference reproduce/paper1/records/round3/geocrack_comparison.json
```

`GEOCRACK` is the folder that holds `Original Images` and `Edge Masks`. Run this way, the script measures the photographs
(about 3 minutes, 2 GB of memory; NumPy, SciPy and Pillow, that is, the labels environment), takes the synthetic side of
the comparison from the stored record and lists every value that differs from the stored record. With `--frames FRAMES`
it also measures the synthetic frames again (the 50 frames of block C of `../frames.csv`, and `a0018` and `a0020`; see
`frames_io.py` for the folder layout), and with `--frames` alone only those. The overlapping coefficient with equal
weight per image needs both.

## FraXet

* **Dataset:** FraXet, version 0.1, Zenodo, <https://doi.org/10.5281/zenodo.17069947>.
* **Licence:** CC BY 4.0. The dataset asks its users to attribute both FraXet and the original sources of the sites
  that are used; they are listed below.
* **References (FraXet):**
  * Fatihi, A., Caldeira, J., Beucler, T., Thiele, S., Samsu, A., 2025. FraXet (version 0.1) [data set]. Zenodo.
    <https://doi.org/10.5281/zenodo.17069947>.
  * Fatihi, A., Caldeira, J., Beucler, T., Thiele, S.T., Samsu, A., 2026. Towards robust fracture mapping: benchmarking
    automatic fracture mapping in 2D outcrop imagery. Solid Earth 17, 1087–1115.
    <https://doi.org/10.5194/se-17-1087-2026>.
* **Original sources, sites OG1 and KB3** (Loviisa coast, Finland; manual traces, and orthomosaics and DEMs):
  * Ovaskainen, N., Nordbäck, N., 2022. Manually mapped traces from UAV-acquired images of Loviisa shoreline outcrops
    (version 1.1.0) [data set]. Zenodo. <https://doi.org/10.5281/zenodo.7077846>.
  * Nordbäck, N., Ovaskainen, N., 2025. UAV-acquired orthomosaics of Loviisa shoreline outcrops [data set]. Zenodo.
    <https://doi.org/10.5281/zenodo.17878870>.
* **Original sources, site EN** (Australia):
  * Samsu, A., Cruden, A., Vollgger, S., 2019. Scale matters: the influence of structural inheritance on fracture
    patterns – Supplementary material [data set]. Monash University. <https://doi.org/10.26180/5cdcad0a73fe0>.
  * Samsu, A., Cruden, A.R., Micklethwaite, S., Grose, L., Vollgger, S.A., 2020. Scale matters: The influence of
    structural inheritance on fracture patterns. Journal of Structural Geology 130, 103896.
    <https://doi.org/10.1016/j.jsg.2019.103896>.
* **How to obtain it:** download the archive of version 0.1 from the Zenodo page (about 2.6 GB) and unpack it. The
  folder `FraXet_v0.1` then holds `ovaskainen23_` and `samsu19`.
* **What is used:** three folders, each with `image/*.png` (orthophoto), `gt/*.tif` (trace annotation) and `dem/*.tif`
  (elevation in metres) in patches of 256 x 256 px:

  | Site | Folder | Patches | Pixel size taken from the literature |
  |---|---|---|---|
  | OG1 | `ovaskainen23_/train` | 2,808 | 5-6 mm |
  | KB3 | `ovaskainen23_/valid` | 966 | 5-6 mm |
  | EN | `samsu19/train` | 642 | 29-32 mm |

```bash
python reproduce/paper1/analysis/fraxet_sites.py --fraxet FRAXET --reference reproduce/paper1/records/round3/fraxet_sites.json
```

`FRAXET` is the folder that holds `ovaskainen23_` and `samsu19`. The script needs NumPy, SciPy, Pillow and PyTorch
(PyTorch is not part of the generator's two environments); the stored record was computed in about 3 minutes with a GPU
on a machine with 64 GB of memory (the largest site is held in memory as one mosaic of 12,800 x 35,584 px; the peak
memory use was not recorded). The synthetic reference of the profile readings is taken from the stored record;
`--frames FRAMES` measures it again from the 50 frames of block C. The record gives the SHA-256 of the sorted patch names of each folder
(`patch_names_sha256`: the names without extension, joined by line feeds), so that the set of patches can be compared.

## What the records hold

Both records start with the dataset, the parameters of the measurement, and notes on how to read the values.

`geocrack_comparison.json`, with the groups `real`, `synthetic_aerial` (50 frames) and `synthetic_close_range` (2 frames):

| Key | Content |
|---|---|
| `groups` | number of images and measurement points, share of points without a dark band, share of the comparable subset |
| `summary.QUANTITY.GROUP.all` / `.comparable` | median, quartiles (`iqr`), 5th and 95th percentile over the measurement points |
| `histograms.QUANTITY` | bin edges and counts of measurement points per group |
| `overlap.QUANTITY.SYNTHETIC_GROUP` | overlapping coefficients real against synthetic: `all_points`, `comparable_subset`, `both_sides_comparable` |
| `coverage.QUANTITY.SYNTHETIC_GROUP` | shares of the real points below, inside and above the synthetic 5th-95th percentile range |
| `per_image` | one row of statistics per photograph and per frame |

`QUANTITY` is one of `surrounding_brightness`, `centre_brightness`, `brightness_difference`, `relative_contrast`,
`dark_band_width_px` and `log10_width_over_spacing` (the logarithm of the paper's relative width; the quantiles of the
ratio itself are under `ratio`), and `background_variation`, which has one value per image. The pooled overlapping
coefficients can be recomputed from the histograms: `sum(min(a / sum(a), b / sum(b)))` over the bins.

`fraxet_sites.json`:

| Key | Content |
|---|---|
| `synthetic_reference` | profile readings of the 50 frames of the multi-illumination aerial set and their weighted histograms |
| `sites.SITE.stitching`, `.mosaic`, `.elevation_grid` | elevation step across patch edges for the two readings of the file names, size of the assembled site, DEM grid |
| `sites.SITE.occlusion.by_pixel_mm.PIXEL` | one row per off-nadir angle and clearance height: occluded, determined and undetermined sample points and their shares, across and along strike |
| `sites.SITE.profiles` | quantiles of the profile readings, dark-band width in pixels and in millimetres, comparison with the synthetic reference (`vs_synthetic`) |
| `sites.SITE.illumination` | regression of brightness on the DEM gradient, estimated azimuth, brightness difference by angle to the trace strike, Spearman correlation |

Ranges in the paper that are written as, for example, "11%-19%" span the two pixel sizes and the two shares
`occluded_share` (undetermined points left out of the denominator) and `occluded_share_undetermined_as_visible`.

## What was checked for this release

The numbers of the paper were computed with internal versions of the two scripts, in which file locations were fixed.
The released scripts take them from the command line, write English keys, and pass the synthetic reference of
`fraxet_sites.py` as weighted histograms; the computations are unchanged. The records are the internal records in the
released layout.

* `geocrack_comparison.py` was run on the 52 frames (every value of the synthetic groups and the six rows of
  `--check-ruler` equal to the record) and, with the command above, on the 49 photographs: all 2836 compared values equal
  the record, and the thinning equals the label program's on the crop of every mask.
* `fraxet_sites.py` reproduced `synthetic_reference` from the 50 frames and was run, with the command above, on the three
  sites. Of the 1715 values computed for the sites, 1713 equal the record within the relative tolerance of 1e-9. The other
  two are the Spearman correlations of the illumination analysis at OG1 and EN, which differ in the fifth and sixth decimal
  (0.09906 against 0.09904, and 0.067994 against 0.067986; the paper gives two decimals); the description at the top of
  the script says why. The three counts `mosaic.annotation_px_outside_valid`, which the internal script did not write, are
  those of this run. Cases that the three sites do not contain (a site whose patches do not fit together, a site whose
  azimuth estimate is not accepted) were compared with the internal script on made-up data of the same folder layout: all
  values were equal, on the GPU and on the CPU.
* Every number of the paper that comes from the two records, including the supplement tables, was recomputed from the
  records and equals the printed value.
