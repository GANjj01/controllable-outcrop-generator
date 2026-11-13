# Provenance of the released code

For the reproduction material of the paper's main text see [the last section](#reproduction-material-of-the-main-text).

The paper's images and labels were produced with an internal working copy of this code. The release differs from it
only as follows; for every Python file listed in the table below, the abstract syntax tree of the released file, with
documentation strings removed, equals that of the internal file after the declared edits.

* comments and documentation strings: notes about the internal workflow (internal rule versions, environments,
  scripts, documents and process notes) were removed or reworded and Chinese text was translated;
* directory names: `multiscene_r1` -> `multiscene`, `rq1_r1` -> `multiscene/driver`, `route_v4` -> `multiscene/host`,
  `net_template` -> `multiscene/cavity_template`, `flat_sets_r2` and `standalone_sets_r1` -> `multiscene/scene_sets`,
  `trace_labels_r2` -> `labels`, the `outcrop` package moved to the repository root; the run-configuration names
  `python.phase0` / `python.cnn` became `python.geometry` / `python.labels`;
* display strings of the geological model (lithology display names, user-interface labels of shape and environment
  parameters) translated from Chinese to English; they do not enter any computation;
* declared code edits (listed per file below) and re-pinned self-checks (hashes of the code's own files);
* data files: internal notes in the text fields of the relief, sky and exposure inputs removed (numbers unchanged); the
  Chinese display-name section of `outcrop/dsgen/data/legacy_names.json` emptied; the hash files of the label
  rules reduced to the hash and the file name (the code reads only the hash); the reference path and the rule name of
  `multiscene/host/edge_ref.json` replaced by descriptions and the description of
  `inputs/labels/lithology_aliases_r1.json` reworded (numbers and tables unchanged).

Re-pinned self-checks:

* `cavity_template/build.py -> floor_relief.py sha256`: `"be4a826b0ab6e5e811e1f5422fd45bf68afd3b8b5a24ee18cfc93ad6e1f14d91"`
* `labels/fault_support.py DEPLOYED_MD5`: `{"build.py": "612dd2317a3e8c4f1bae05204b19ee30", "prepare.py": "7fecbde7bb2e7fcf20c96f5555e32852", "frgb/fault_charts.py": "61ee4c25f208cf65c0095ec816a9f145"}`
* `labels/fault_support.py DEPLOYED_NETWORK_PANELS_SHA256`: `"68ec33412bbb2e1a9be254d4d53c83017008a31866f721fc379b371721c7ef6e"`
* `multiscene/driver/label_cached.py PINNED / PINNED_R4`: `["22431c04ced9c9decbe8844e40bf19f19f27ace41c79d2ee271fa265048c2a8d", "4cc934ea35595e513c536874f4e2ace819ee0c8712c683e2f62e4ce3f72db3ca"]`

| Released file | Internal file | Internal sha256 (first 12) | Released sha256 (first 12) | Declared edits |
|---|---|---|---|---|
| `frgb/budget.py` | `frgb/budget.py` | 45ae9f73d6f9 | 70fcaed97275 |  |
| `frgb/cd_search.py` | `frgb/cd_search.py` | 2594e2117b63 | e60d6768481d |  |
| `frgb/chain/adaptive_refine.py` | `frgb/chain/adaptive_refine.py` | 08de7318d50f | 8e34c2549bef |  |
| `frgb/chain/blender_client.py` | `frgb/chain/blender_client.py` | 816ed9164d6c | edb18dbadc0f |  |
| `frgb/chain/blender_stage.py` | `frgb/chain/blender_stage.py` | 86844538ca6d | 28a354f144ce |  |
| `frgb/chain/bluff_exact.py` | `frgb/chain/bluff_exact.py` | 58e35e4e5ac4 | 4925ff49ac03 |  |
| `frgb/chain/bluff_refine.py` | `frgb/chain/bluff_refine.py` | 5c8570d475d6 | edefd80382d2 |  |
| `frgb/chain/bluff_reproduce.py` | `frgb/chain/bluff_reproduce.py` | 0ea4c1a6ca87 | 1d5c8123c56b |  |
| `frgb/chain/evaluate_lines.py` | `frgb/chain/evaluate_lines.py` | 526164bf269e | 526164bf269e |  |
| `frgb/chain/make_source_raster.py` | `frgb/chain/make_source_raster.py` | e43b5eb64961 | 37bfb3a1b8ef |  |
| `frgb/chain/run_frames.py` | `frgb/chain/run_frames.py` | 1dc10a41c149 | 8ee9d4468ee2 |  |
| `frgb/chain/sliver_render.py` | `frgb/chain/sliver_render.py` | 14fb6481b53f | 0ee140d17538 |  |
| `frgb/chain/worker.py` | `frgb/chain/worker.py` | b372eebbe9b9 | 35d3062327eb |  |
| `frgb/difficulty_params.py` | `frgb/difficulty_params.py` | 6fa7b779f7f9 | dc9cfc86af53 | record string (status note) reworded |
| `frgb/fault_charts.py` | `frgb/fault_charts.py` | f3a2e1c15a9c | 6bfc835c6d85 |  |
| `frgb/fault_traces.py` | `frgb/fault_traces.py` | ba12b0dd7956 | 4496841f4ac4 |  |
| `frgb/gate.py` | `frgb/gate.py` | 43fe35e92f78 | 52f05c7cfa97 |  |
| `frgb/lighting_rules.py` | `frgb/lighting_rules.py` | 2c7a47c56446 | aa5ecf45b236 |  |
| `frgb/lith_records.py` | `frgb/lith_records.py` | d80dde97d136 | cf3ff643da83 |  |
| `frgb/metrics.py` | `frgb/metrics.py` | 45a9e6fe8201 | 22be8561ce21 |  |
| `frgb/morphology.py` | `frgb/morphology.py` | 26f52a4d3ba2 | 26f52a4d3ba2 |  |
| `frgb/network_cutters.py` | `frgb/network_cutters.py` | 0046412861b6 | a66f3a696f2a |  |
| `frgb/network_panels.py` | `frgb/network_panels.py` | 6e0272773da0 | 68ec33412bbb |  |
| `frgb/network_profiles.py` | `frgb/network_profiles.py` | af0adecb230b | 6b255f26ed05 |  |
| `frgb/network_training_labels.py` | `frgb/network_training_labels.py` | 5007b4f77168 | 68f9b51bacfe |  |
| `frgb/raster.py` | `frgb/raster.py` | 15872da853e1 | 649c80bb150b |  |
| `frgb/rq1_design.py` | `frgb/rq1_design.py` | 39b3d8f02b1c | 0b02ecfcc4b5 | record strings (status notes) reworded |
| `frgb/rq1_multiscene.py` | `frgb/rq1_multiscene.py` | 7d36981aca41 | c8eef849a7db | error message of train_round reworded |
| `frgb/runconfig.py` | `frgb/runconfig.py` | aada350e6fe3 | 60340af01664 |  |
| `frgb/scene_light.py` | `frgb/scene_light.py` | 0cf7273cac01 | 845a2729505e |  |
| `frgb/scene_relief.py` | `frgb/scene_relief.py` | 4740b75e99de | b85031013213 | record strings (status notes of the relief specifications) reworded |
| `frgb/trace_labels.py` | `frgb/trace_labels.py` | 211bbc3dc720 | 738d527a4586 |  |
| `frgb/trace_targets.py` | `frgb/trace_targets.py` | 92edff36079f | d536d9acc751 |  |
| `frgb/tracechains.py` | `frgb/tracechains.py` | a9d017d24125 | a9d017d24125 |  |
| `inputs/exposure/exposure_gate_r14.json` | `inputs/exposure/exposure_gate_r14.json` | 88001a219018 | 42a16d146cb8 |  |
| `inputs/labels/lithology_aliases_r1.json` | `inputs/labels/lithology_aliases_r1.json` | e9ba23511308 | a33e39e42def | description reworded (tables unchanged) |
| `inputs/relief/self_affine_spec_C_r2.json` | `inputs/relief/self_affine_spec_C_r2.json` | 669d29bfd14b | 2773d58c674a |  |
| `inputs/relief/wall_relief_dev_t02_C_r2.json` | `inputs/relief/wall_relief_dev_t02_C_r2.json` | 5270bffa2197 | 5698e3e1abda |  |
| `inputs/render/sky_calibration.json` | `inputs/render/sky_calibration.json` | c4ce9221a633 | 084cd9b3bf5e |  |
| `labels/criterion_trace_labels_r2.sha256.txt` | `trace_labels_r2/criterion_trace_labels_r2.sha256.txt` | 8d3fbced091f | 0291b5fe2c13 |  |
| `labels/criterion_trace_labels_r2_amendment_a.sha256.txt` | `trace_labels_r2/criterion_trace_labels_r2_amendment_a.sha256.txt` | 6bea1398f976 | c73a303580ed |  |
| `labels/criterion_trace_labels_r3.sha256.txt` | `trace_labels_r2/criterion_trace_labels_r3.sha256.txt` | 2da3a71d2209 | 327e2d260490 |  |
| `labels/criterion_trace_labels_r3_amendment_a.sha256.txt` | `trace_labels_r2/criterion_trace_labels_r3_amendment_a.sha256.txt` | 75c76a3ac933 | bfd54ca1468f |  |
| `labels/criterion_trace_labels_r3_amendment_b.sha256.txt` | `trace_labels_r2/criterion_trace_labels_r3_amendment_b.sha256.txt` | 918f755630a1 | bfb8523da560 |  |
| `labels/criterion_trace_labels_r3_amendment_c.sha256.txt` | `trace_labels_r2/criterion_trace_labels_r3_amendment_c.sha256.txt` | 66633af69930 | f13786ffd7bd |  |
| `labels/criterion_trace_labels_r4_faults.sha256.txt` | `trace_labels_r2/criterion_trace_labels_r4_faults.sha256.txt` | 87d1fb182a92 | 2c2243209273 |  |
| `labels/fault_sides.py` | `trace_labels_r2/fault_sides.py` | 3b45f16779e7 | e0314883c7d5 |  |
| `labels/fault_support.py` | `trace_labels_r2/fault_support.py` | da277b42a703 | ddf699c65c44 | internal builder hashes kept as INTERNAL_MD5 and INTERNAL_NETWORK_PANELS_SHA256; an unused constant and an unused helper function removed |
| `labels/label_frame.py` | `trace_labels_r2/label_frame.py` | 7d9df3af1c14 | 22431c04ced9 |  |
| `labels/label_frame_r4.py` | `trace_labels_r2/label_frame_r4.py` | c80fe6a92942 | 4cc934ea3559 |  |
| `labels/mismatch.py` | `trace_labels_r2/mismatch.py` | 87ed0fd552f7 | 2b8f0573d8d2 |  |
| `labels/occlusion_rule.py` | `trace_labels_r2/occlusion_rule.py` | 82aff61ad8b7 | dc86258ab057 |  |
| `labels/procmem.py` | `trace_labels_r2/procmem.py` | 84a7281aab26 | 0542ea4a3fca |  |
| `labels/rebuild_cavities.py` | `trace_labels_r2/rebuild_cavities.py` | dbf124a5a167 | fffc00325d09 | default scratch directory is the temporary directory; the builder-file check accepts the released and the internal builder files |
| `labels/run_frames_r2.py` | `trace_labels_r2/run_frames_r2.py` | eefa86cbc4af | 07c9a343e7d3 | internal validation groups (stored outside the generator) no longer loaded at import |
| `labels/run_frames_r3.py` | `trace_labels_r2/run_frames_r3.py` | 0576c30337d8 | 4e39bf81b916 |  |
| `multiscene/camera_plan.py` | `multiscene_r1/camera_plan.py` | c0df2a50c36a | cc79fab20e84 |  |
| `multiscene/camera_plan_r2.py` | `multiscene_r1/camera_plan_r2.py` | 854a805a02b9 | ddaa72bb5ed9 | record strings (criteria, reason of a skipped cell) reworded |
| `multiscene/cavity_template/build.py` | `multiscene_r1/net_template/build.py` | 5394a7b40cc3 | e7ceab94a0b2 |  |
| `multiscene/cavity_template/check_host_provenance.py` | `multiscene_r1/net_template/check_host_provenance.py` | 4d347f31f6f9 | 161cda72643e |  |
| `multiscene/cavity_template/export.py` | `multiscene_r1/net_template/export.py` | ae0b693f8151 | 82b28d1f19c0 |  |
| `multiscene/cavity_template/floor_relief.py` | `multiscene_r1/net_template/floor_relief.py` | ec622ea7e43f | be4a826b0ab6 |  |
| `multiscene/cavity_template/prepare.py` | `multiscene_r1/net_template/prepare.py` | f4054dd39f99 | dbfe7174fa6a |  |
| `multiscene/cavity_template/repair_host_needles.py` | `multiscene_r1/net_template/repair_host_needles.py` | dd530dd84897 | e46db0c15796 | record string (rule tag) reworded |
| `multiscene/cavity_template/source_tables.py` | `multiscene_r1/net_template/source_tables.py` | d5f4837f4e7c | db53e1e20bce |  |
| `multiscene/driver/export_norim.py` | `multiscene_r1/rq1_r1/export_norim.py` | 41c7acce055e | 10df57046603 |  |
| `multiscene/driver/label_cached.py` | `multiscene_r1/rq1_r1/label_cached.py` | 3890977e9077 | 98a0348da8f2 |  |
| `multiscene/driver/render_server.py` | `multiscene_r1/rq1_r1/render_server.py` | 519ba876ecc3 | d26d24559c65 | render ledger optional (render spec ledger: null) |
| `multiscene/driver/run_rq1.py` | `multiscene_r1/rq1_r1/run_rq1.py` | 4d15aef44d03 | 6ebd1cb7ff3a | default host-route directory renamed; messages and record strings reworded (lock text; wait, reservation and memory-prediction messages; reservation and exposure rule notes; path check) |
| `multiscene/field_display.py` | `multiscene_r1/field_display.py` | 097d7901a732 | 46c8988d78ff |  |
| `multiscene/host/dsgen_overlay/processes_gt4.py` | `multiscene_r1/route_v4/dsgen_overlay/processes_gt4.py` | 46538042fb21 | 03539c7d102c |  |
| `multiscene/host/edge_ref.json` | `multiscene_r1/route_v4/edge_ref.json` | d1c324bb863a | 2621722afe9a | reference path and rule name replaced by descriptions (numbers unchanged) |
| `multiscene/host/exact_route.py` | `multiscene_r1/route_v4/exact_route.py` | 31ca7765e6e1 | 4c3b1b3e6f85 |  |
| `multiscene/host/fracture_density.py` | `multiscene_r1/route_v4/fracture_density.py` | 6df85b6c971a | 1fefeb00acdc |  |
| `multiscene/host/gen_normal4.py` | `multiscene_r1/route_v4/gen_normal4.py` | 4ed4d000007e | 4d2cace593fd |  |
| `multiscene/host/groove_tip.py` | `multiscene_r1/route_v4/groove_tip.py` | f4feca73e99e | 9006988ee150 |  |
| `multiscene/host/n_field_normals.py` | `multiscene_r1/route_v4/n_field_normals.py` | a719e4bbddc9 | f43a99a4dc81 |  |
| `multiscene/host/n_zone_levels.py` | `multiscene_r1/route_v4/n_zone_levels.py` | f99d79c16333 | 4dfafd663c6f |  |
| `multiscene/host/n_zone_refine.py` | `multiscene_r1/route_v4/n_zone_refine.py` | dcc7f7a54bfa | ae6621ebf583 | record string (base) reworded |
| `multiscene/host/n_zone_shape_check.py` | `multiscene_r1/route_v4/n_zone_shape_check.py` | 581b82363067 | 20fbbe037d9b |  |
| `multiscene/host/nogpu.py` | `multiscene_r1/route_v4/nogpu.py` | 236a779ef152 | b8dc5e068ab3 |  |
| `multiscene/host/normal_exact.py` | `multiscene_r1/route_v4/normal_exact.py` | b3aa19adf2b8 | 15cff9732de4 |  |
| `multiscene/host/normal_surface.py` | `multiscene_r1/route_v4/normal_surface.py` | a18f71701fcc | 0c53e7afefe6 | record string (construction note) reworded |
| `multiscene/host/overlay4.py` | `multiscene_r1/route_v4/overlay4.py` | ef9cd6cd0762 | 1e2aa1133dbf |  |
| `multiscene/light_predict.py` | `multiscene_r1/light_predict.py` | b592ff794c98 | f27fd1321171 |  |
| `multiscene/scene_generate.py` | `multiscene_r1/scene_generate.py` | 87ef8448ebfc | 995364857c41 |  |
| `multiscene/scene_sets/fs_camera_plan.py` | `multiscene_r1/flat_sets_r2/fs_camera_plan.py` | cf6d0c675c85 | 4e96da2ba19e | record string (reason) reworded |
| `multiscene/scene_sets/fs_effective_config.py` | `multiscene_r1/flat_sets_r2/fs_effective_config.py` | d31798369ebd | e730c46763e4 |  |
| `multiscene/scene_sets/fs_nfaults.py` | `multiscene_r1/flat_sets_r2/fs_nfaults.py` | 7130a405f897 | abbcfe50a977 |  |
| `multiscene/scene_sets/nogpu.py` | `multiscene_r1/flat_sets_r2/nogpu.py` | 236a779ef152 | b8dc5e068ab3 |  |
| `multiscene/scene_sets/nogpu.py` | `multiscene_r1/standalone_sets_r1/nogpu.py` | 236a779ef152 | b8dc5e068ab3 |  |
| `multiscene/scene_sets/si_camera_plan.py` | `multiscene_r1/standalone_sets_r1/si_camera_plan.py` | 480c45316500 | 8201f6949753 | record string (reason) reworded |
| `multiscene/zone_dist.py` | `multiscene_r1/zone_dist.py` | 2146b61a7d01 | 3815d669af42 |  |
| `multiscene/zone_raster_check.py` | `multiscene_r1/zone_raster_check.py` | 95b8e61bc727 | 7c260ffebd59 |  |
| `outcrop/__init__.py` | `repository src/outcrop/__init__.py` | 2a0b0b3984f6 | 2a0b0b3984f6 |  |
| `outcrop/contracts.py` | `repository src/outcrop/contracts.py` | 15cadd8e2d89 | 15cadd8e2d89 |  |
| `outcrop/dataset/__init__.py` | `repository src/outcrop/dataset/__init__.py` | e8420c2aa482 | e8420c2aa482 |  |
| `outcrop/dataset/export.py` | `repository src/outcrop/dataset/export.py` | 07170ae9e521 | 07170ae9e521 |  |
| `outcrop/dsgen/__init__.py` | `repository src/outcrop/dsgen/__init__.py` | 284025c96e97 | 6341fc9c5c52 |  |
| `outcrop/dsgen/blockface.py` | `repository src/outcrop/dsgen/blockface.py` | 18d95bf5cfad | df0af2223430 |  |
| `outcrop/dsgen/boundary.py` | `repository src/outcrop/dsgen/boundary.py` | 261f5e0e7620 | 261f5e0e7620 |  |
| `outcrop/dsgen/canyon.py` | `repository src/outcrop/dsgen/canyon.py` | 051d84070bca | cc40619e936e |  |
| `outcrop/dsgen/cmgeom.py` | `repository src/outcrop/dsgen/cmgeom.py` | 72f7608abb87 | f7bf76d04c33 |  |
| `outcrop/dsgen/coast.py` | `repository src/outcrop/dsgen/coast.py` | 23baf28563f4 | 02642fb81f25 |  |
| `outcrop/dsgen/data/legacy_names.json` | `repository src/outcrop/dsgen/data/legacy_names.json` | 68a29127b387 | 270a278226dc |  |
| `outcrop/dsgen/declared.py` | `repository src/outcrop/dsgen/declared.py` | b03434ea80a0 | 3c2b807dd73c |  |
| `outcrop/dsgen/display.py` | `repository src/outcrop/dsgen/display.py` | fcd9f124d428 | 54933a5b29a4 |  |
| `outcrop/dsgen/env/__init__.py` | `repository src/outcrop/dsgen/env/__init__.py` | 9272d87c8736 | 67d6f869a9d3 |  |
| `outcrop/dsgen/env/catalog.py` | `repository src/outcrop/dsgen/env/catalog.py` | 29d06c156974 | ee847a1cc032 |  |
| `outcrop/dsgen/env/config.py` | `repository src/outcrop/dsgen/env/config.py` | eae79768fa15 | b9249ca9513c |  |
| `outcrop/dsgen/environment.py` | `repository src/outcrop/dsgen/environment.py` | c9f4bce6004d | e646128b9097 |  |
| `outcrop/dsgen/exposure.py` | `repository src/outcrop/dsgen/exposure.py` | 8553fa0da766 | b84b8423a55f |  |
| `outcrop/dsgen/faults.py` | `repository src/outcrop/dsgen/faults.py` | b2f55761717a | 9a56d9a8a817 |  |
| `outcrop/dsgen/fracture.py` | `repository src/outcrop/dsgen/fracture.py` | e87fc57d77c8 | c1eac30d3b6e |  |
| `outcrop/dsgen/fracture_network.py` | `repository src/outcrop/dsgen/fracture_network.py` | b9283c27623e | f870eb5e7e2f |  |
| `outcrop/dsgen/generation_config.py` | `repository src/outcrop/dsgen/generation_config.py` | ceec6ef19577 | 07782cbbcd9d | appearance provider list reduced to manual (an unused external texture service removed) |
| `outcrop/dsgen/labels.py` | `repository src/outcrop/dsgen/labels.py` | bda7d8c0b859 | 661167fbb5b7 |  |
| `outcrop/dsgen/large_cliff.py` | `repository src/outcrop/dsgen/large_cliff.py` | e1a3e4d0a112 | e1a3e4d0a112 |  |
| `outcrop/dsgen/legacy.py` | `repository src/outcrop/dsgen/legacy.py` | 933dc40f29fc | 332461243fab |  |
| `outcrop/dsgen/lighting.py` | `repository src/outcrop/dsgen/lighting.py` | 09b6d0840e2a | acdde074de57 |  |
| `outcrop/dsgen/lithology.py` | `repository src/outcrop/dsgen/lithology.py` | 53602d0e846c | 5ad0a3a44509 |  |
| `outcrop/dsgen/morph.py` | `repository src/outcrop/dsgen/morph.py` | 395e7cdef333 | b9aa27ea3590 |  |
| `outcrop/dsgen/paper_geology.py` | `repository src/outcrop/dsgen/paper_geology.py` | 774762c62070 | 9d9f76322fd3 |  |
| `outcrop/dsgen/pipeline.py` | `repository src/outcrop/dsgen/pipeline.py` | 80bffd046f8d | e14c94baad13 |  |
| `outcrop/dsgen/processes.py` | `repository src/outcrop/dsgen/processes.py` | b5220b5a2b9b | b1f93d6eaa18 |  |
| `outcrop/dsgen/profiles.py` | `repository src/outcrop/dsgen/profiles.py` | bb031e31f04a | bb031e31f04a |  |
| `outcrop/dsgen/relief.py` | `repository src/outcrop/dsgen/relief.py` | 6d1360ba68e5 | 9f5e38e638d8 |  |
| `outcrop/dsgen/roadcut.py` | `repository src/outcrop/dsgen/roadcut.py` | 3467cf468034 | 567026d4668b |  |
| `outcrop/dsgen/scandetail.py` | `repository src/outcrop/dsgen/scandetail.py` | ce878685dc18 | 370fe9d82033 |  |
| `outcrop/dsgen/scanmesh.py` | `repository src/outcrop/dsgen/scanmesh.py` | 4940eb3a0eb4 | 09e50ea3577f |  |
| `outcrop/dsgen/scanned_scenes.py` | `repository src/outcrop/dsgen/scanned_scenes.py` | 2880e78680ed | 806f4b45e704 | paths of external scanned-scene projects removed (preset names kept) |
| `outcrop/dsgen/scene.py` | `repository src/outcrop/dsgen/scene.py` | b5d3ae229f5d | f706b212ce01 |  |
| `outcrop/dsgen/scene_config.py` | `repository src/outcrop/dsgen/scene_config.py` | bea57276f0e4 | 2e2fbd7716d4 |  |
| `outcrop/dsgen/semantics.py` | `repository src/outcrop/dsgen/semantics.py` | f7da9f16bd0d | 6f2a74a54318 |  |
| `outcrop/dsgen/shape_config.py` | `repository src/outcrop/dsgen/shape_config.py` | 249aee80228e | f411dc2220ae |  |
| `outcrop/dsgen/sky.py` | `repository src/outcrop/dsgen/sky.py` | da8a7e5e5155 | 46165b45663e |  |
| `outcrop/dsgen/stage.py` | `repository src/outcrop/dsgen/stage.py` | 73d8eb343e12 | 42c1755f6b4c | machine paths of the game-engine staging (not used by the generator) made relative |
| `outcrop/dsgen/stratigraphy.py` | `repository src/outcrop/dsgen/stratigraphy.py` | 23dc99dba4f7 | 7b73422a7756 |  |
| `outcrop/dsgen/surface.py` | `repository src/outcrop/dsgen/surface.py` | e271cedfce0d | 400cd6d58e42 |  |
| `outcrop/dsgen/truth.py` | `repository src/outcrop/dsgen/truth.py` | 2aba8dc9a7fd | 76dc6eba1bda |  |
| `outcrop/dsgen/wall_evidence.py` | `repository src/outcrop/dsgen/wall_evidence.py` | 62faa5c13986 | 07ccdbac0d4b |  |
| `outcrop/dsgen/wallcheck.py` | `repository src/outcrop/dsgen/wallcheck.py` | 413c56ab1652 | d2d1d3c79b25 |  |
| `outcrop/dsgen/wallscene.py` | `repository src/outcrop/dsgen/wallscene.py` | 67d6a3aefbe6 | c90223dc8d26 |  |
| `outcrop/dsgen/xfermat.py` | `repository src/outcrop/dsgen/xfermat.py` | 7ef9fb64487b | e821605c8571 |  |
| `outcrop/gpu_runtime.py` | `repository src/outcrop/gpu_runtime.py` | 0a8dbd0c52a2 | 9dacc4e78511 | GPU cache directory from FRGB_GPU_CACHE or the temporary directory instead of a workspace storage map; skipped when CuPy is disabled |
| `outcrop/modeling/__init__.py` | `repository src/outcrop/modeling/__init__.py` | 736791fdad93 | 736791fdad93 |  |
| `outcrop/modeling/body_fields.py` | `repository src/outcrop/modeling/body_fields.py` | 99c8217d1833 | 99c8217d1833 |  |
| `outcrop/modeling/compiler.py` | `repository src/outcrop/modeling/compiler.py` | db17e6c3b8cb | db17e6c3b8cb |  |
| `outcrop/modeling/constraints.py` | `repository src/outcrop/modeling/constraints.py` | b16cc0eef296 | b16cc0eef296 |  |
| `outcrop/modeling/decode.py` | `repository src/outcrop/modeling/decode.py` | 058ef1e43486 | 827084fef435 |  |
| `outcrop/modeling/deformations.py` | `repository src/outcrop/modeling/deformations.py` | f9bd548c073d | f9bd548c073d |  |
| `outcrop/modeling/digests.py` | `repository src/outcrop/modeling/digests.py` | c8227ccc0264 | c8227ccc0264 |  |
| `outcrop/modeling/errors.py` | `repository src/outcrop/modeling/errors.py` | 54d0e9ef2af1 | 54d0e9ef2af1 |  |
| `outcrop/modeling/events.py` | `repository src/outcrop/modeling/events.py` | 5e6dc2740ca8 | 5e6dc2740ca8 |  |
| `outcrop/modeling/exposure.py` | `repository src/outcrop/modeling/exposure.py` | 9eb24407ddbf | 9eb24407ddbf |  |
| `outcrop/modeling/fabrics.py` | `repository src/outcrop/modeling/fabrics.py` | c6990014c126 | c6990014c126 |  |
| `outcrop/modeling/faults.py` | `repository src/outcrop/modeling/faults.py` | fceb8e917694 | 80aa60590463 |  |
| `outcrop/modeling/features.py` | `repository src/outcrop/modeling/features.py` | f19c825406ba | f19c825406ba |  |
| `outcrop/modeling/fractures.py` | `repository src/outcrop/modeling/fractures.py` | 1c82acb30d43 | 1c82acb30d43 |  |
| `outcrop/modeling/groove_traces.py` | `repository src/outcrop/modeling/groove_traces.py` | 6d8cc04aca6f | 6d8cc04aca6f |  |
| `outcrop/modeling/intrusions.py` | `repository src/outcrop/modeling/intrusions.py` | 97d33bcdf211 | 97d33bcdf211 |  |
| `outcrop/modeling/joints.py` | `repository src/outcrop/modeling/joints.py` | be348a3ded5b | be348a3ded5b |  |
| `outcrop/modeling/kernel/__init__.py` | `repository src/outcrop/modeling/kernel/__init__.py` | 48fbd12e5cb4 | 48fbd12e5cb4 |  |
| `outcrop/modeling/kernel/_exact_native.cpp` | `repository src/outcrop/modeling/kernel/_exact_native.cpp` | 9424fa3820f5 | 9424fa3820f5 |  |
| `outcrop/modeling/kernel/analytic.py` | `repository src/outcrop/modeling/kernel/analytic.py` | 75d86ec83e8d | 861ab1b64102 |  |
| `outcrop/modeling/kernel/arrangement.py` | `repository src/outcrop/modeling/kernel/arrangement.py` | fd82d58207be | fd82d58207be |  |
| `outcrop/modeling/kernel/boundary.py` | `repository src/outcrop/modeling/kernel/boundary.py` | 441370e7b490 | e0dc7a2900b3 |  |
| `outcrop/modeling/kernel/certified_arithmetic.py` | `repository src/outcrop/modeling/kernel/certified_arithmetic.py` | 1758184b2227 | 1758184b2227 |  |
| `outcrop/modeling/kernel/csg.py` | `repository src/outcrop/modeling/kernel/csg.py` | fc75bed212e0 | fc75bed212e0 |  |
| `outcrop/modeling/kernel/entities.py` | `repository src/outcrop/modeling/kernel/entities.py` | 8d9e86d311a8 | 8d9e86d311a8 |  |
| `outcrop/modeling/kernel/error_bounds.py` | `repository src/outcrop/modeling/kernel/error_bounds.py` | 8e808c6e050c | 8e808c6e050c |  |
| `outcrop/modeling/kernel/exact_fields.py` | `repository src/outcrop/modeling/kernel/exact_fields.py` | 0542889d1ac9 | 0542889d1ac9 |  |
| `outcrop/modeling/kernel/facet_filter.py` | `repository src/outcrop/modeling/kernel/facet_filter.py` | 1288d293bbea | 1288d293bbea |  |
| `outcrop/modeling/kernel/gpu_filters.py` | `repository src/outcrop/modeling/kernel/gpu_filters.py` | 01e0e18a0a2f | 01e0e18a0a2f |  |
| `outcrop/modeling/kernel/grid.py` | `repository src/outcrop/modeling/kernel/grid.py` | 2546606b500a | 2546606b500a |  |
| `outcrop/modeling/kernel/intersections.py` | `repository src/outcrop/modeling/kernel/intersections.py` | f5fcb9a10f54 | f5fcb9a10f54 |  |
| `outcrop/modeling/kernel/map_certification.py` | `repository src/outcrop/modeling/kernel/map_certification.py` | ffc4287dafb0 | ffc4287dafb0 |  |
| `outcrop/modeling/kernel/mapped.py` | `repository src/outcrop/modeling/kernel/mapped.py` | 3234aeeb8633 | 3234aeeb8633 |  |
| `outcrop/modeling/kernel/mapped_surfaces.py` | `repository src/outcrop/modeling/kernel/mapped_surfaces.py` | 56b277548d39 | 56b277548d39 |  |
| `outcrop/modeling/kernel/maps.py` | `repository src/outcrop/modeling/kernel/maps.py` | 85c34ed97c32 | 85c34ed97c32 |  |
| `outcrop/modeling/kernel/mesh.py` | `repository src/outcrop/modeling/kernel/mesh.py` | ca0644d8ff59 | ca0644d8ff59 |  |
| `outcrop/modeling/kernel/native.py` | `repository src/outcrop/modeling/kernel/native.py` | 14f87f910398 | 14f87f910398 |  |
| `outcrop/modeling/kernel/partition.py` | `repository src/outcrop/modeling/kernel/partition.py` | a99c3f273471 | d083cef99f42 |  |
| `outcrop/modeling/kernel/polytopes.py` | `repository src/outcrop/modeling/kernel/polytopes.py` | e3ec7dc9e99c | 4009c9306546 |  |
| `outcrop/modeling/kernel/predicates.py` | `repository src/outcrop/modeling/kernel/predicates.py` | 67c375f1f322 | 4865244b76ef |  |
| `outcrop/modeling/kernel/proxies.py` | `repository src/outcrop/modeling/kernel/proxies.py` | 175854c15002 | 1629200d5d0e |  |
| `outcrop/modeling/kernel/sets.py` | `repository src/outcrop/modeling/kernel/sets.py` | e17a28922f1d | e17a28922f1d |  |
| `outcrop/modeling/kernel/solid_decomposition.py` | `repository src/outcrop/modeling/kernel/solid_decomposition.py` | d030f680848c | d030f680848c |  |
| `outcrop/modeling/kernel/spatial.py` | `repository src/outcrop/modeling/kernel/spatial.py` | d2eb5e8c1837 | d2eb5e8c1837 |  |
| `outcrop/modeling/kernel/surface_arrangement.py` | `repository src/outcrop/modeling/kernel/surface_arrangement.py` | c6a737f40ea0 | c6a737f40ea0 |  |
| `outcrop/modeling/kernel/sweeps.py` | `repository src/outcrop/modeling/kernel/sweeps.py` | fb8b0650a306 | fb8b0650a306 |  |
| `outcrop/modeling/kernel/types.py` | `repository src/outcrop/modeling/kernel/types.py` | 5c4d5faf592e | 5c4d5faf592e |  |
| `outcrop/modeling/kinematics.py` | `repository src/outcrop/modeling/kinematics.py` | 918bef4d8a80 | c04ef2d20d02 |  |
| `outcrop/modeling/labels.py` | `repository src/outcrop/modeling/labels.py` | 4473bde402e3 | 4473bde402e3 |  |
| `outcrop/modeling/materials.py` | `repository src/outcrop/modeling/materials.py` | 4befafe8916c | 4befafe8916c |  |
| `outcrop/modeling/mesh_contracts.py` | `repository src/outcrop/modeling/mesh_contracts.py` | 5c4af7156e1f | 5c4af7156e1f |  |
| `outcrop/modeling/meshing.py` | `repository src/outcrop/modeling/meshing.py` | f8defe27444e | cfde11416052 |  |
| `outcrop/modeling/package.py` | `repository src/outcrop/modeling/package.py` | 66a25f4e2838 | d98a272301e1 |  |
| `outcrop/modeling/parameters.py` | `repository src/outcrop/modeling/parameters.py` | bf03876c7ad6 | bf03876c7ad6 |  |
| `outcrop/modeling/query.py` | `repository src/outcrop/modeling/query.py` | 863e3ec57c94 | 863e3ec57c94 |  |
| `outcrop/modeling/relations.py` | `repository src/outcrop/modeling/relations.py` | 984b83037e98 | 984b83037e98 |  |
| `outcrop/modeling/resolution.py` | `repository src/outcrop/modeling/resolution.py` | 81271511a17b | 81271511a17b |  |
| `outcrop/modeling/resolved.py` | `repository src/outcrop/modeling/resolved.py` | 4380d9d8a511 | 4380d9d8a511 |  |
| `outcrop/modeling/resources.py` | `repository src/outcrop/modeling/resources.py` | 3a0dba302293 | 3a0dba302293 |  |
| `outcrop/modeling/sampling.py` | `repository src/outcrop/modeling/sampling.py` | efecf2bf9a6c | efecf2bf9a6c |  |
| `outcrop/modeling/schema.py` | `repository src/outcrop/modeling/schema.py` | 68d9f415b067 | 68d9f415b067 |  |
| `outcrop/modeling/termination.py` | `repository src/outcrop/modeling/termination.py` | 228fb9465d3a | 228fb9465d3a |  |
| `outcrop/modeling/traces.py` | `repository src/outcrop/modeling/traces.py` | 7bd4d91d0cbf | 7bd4d91d0cbf |  |
| `outcrop/modeling/validation.py` | `repository src/outcrop/modeling/validation.py` | 9fd9dc5b80ce | 9fd9dc5b80ce |  |
| `outcrop/validation.py` | `repository src/outcrop/validation.py` | 9cf16fb2b145 | 6ac32c0451c2 |  |

## Reproduction material of the main text

Besides the per-frame records and `reproduce/paper1/tables.py`, which recomputes the tables of supplement
S1-S3, `reproduce/paper1/` holds the material behind the main text of the paper (Sections 4 and 5, Table 2, Figs 4 to 6):

* `reproduce/paper1/records/round3/`: 11 JSON records and the window table of the illumination sweep;
* `reproduce/paper1/analysis/`: the 12 scripts behind these records and their helper module `frames_io.py`, with two README
  files;
* `reproduce/paper1/figures/`: the scripts that draw Figs 4, 5 and 6;
* `reproduce/paper1/sweeps.csv` and `sweeps.json`: the frames of the view-angle sweep and of the illumination sweep;
* `reproduce/paper1/main_text.py` and `main_text_values.json`: recomputation of the numbers of the main text, with two
  tests in `tests/test_repository.py`.

The generator does not import these scripts. The frames of the two sweeps of Section 5 were generated with an earlier state of
this release, which differed from the released files only in comments, documentation strings (a few scripts copy theirs into a
record as `method`), messages and record strings, descriptions in three input files, and the two unused definitions of
`labels/fault_support.py` that were removed afterwards; none of these enters an array.

The scripts are the analysis scripts used for the paper with these edits: file locations are given on the command line
instead of being fixed, keys and text are in English with the terms of the main README, and notes about the internal workflow
were removed. The records are the records behind the paper in that layout, with unchanged numbers; a few counts that the
paper quotes were added to them so that the quoted value can be recomputed. Each record of the synthetic experiments was
written by its released script and compared with the record behind the paper (every shared number equal). For the two
comparisons with real data, what was run again is stated in `reproduce/paper1/analysis/README_real_data.md`.

The records in `reproduce/paper1/records/` outside `round3/` were written by the analysis scripts of the paper's runs, which are
not part of this package, from the frames, labels and step records of those runs.
Their numbers, keys, and frame and case identifiers are unchanged; the descriptive texts (`rule`, `decl_basis`, notes) and two
recorded error messages of `values_relief_formal.json` were rewritten for the release.
