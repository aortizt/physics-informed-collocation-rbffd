# Collocation and Green Functions for Elliptic Interface Problems

Reproducibility code and archived numerical evidence for the manuscript

> **Benefits and limits of physics-informed point allocation for elliptic interface problems: Controlled RBF--FD, adjoint, and Robin-coupling comparisons**

Arturo Ortiz-Tapia and MartÃ­n Alberto DÃ­az-Viera  
*Journal of Computational Physics submission materials*

## Purpose and scope

This repository documents the computational results reported in the accompanying manuscript. It studies when physics-informed point allocation is useful for stationary elliptic interface problems, and when its computational cost or numerical safeguards limit that usefulness.

The experiments compare protected Green--Halton--Gradient (GHG) point allocation with Halton, residual-adaptive, variation-adaptive, quasi-uniform, defect-only, adjoint-only, and hybrid GHG--adjoint controls. The numerical settings include PHS-RBF-FD interface discretizations, Robin-coupled circular inclusions, controlled smooth and localized-layer cases, and an SPE10 Layer 68 allocation study using P1 finite elements.

The repository supports three distinct reproducibility tasks:

1. **Regenerate reported analyses and figures** from the supplied CSV/NPZ evidence.
2. **Audit recorded production archives** against their manifests, protocols, saved states, and case records.
3. **Rerun numerical campaigns** from the released drivers and original input data.

These are deliberately different tasks. Regenerating a figure does not rerun a PDE campaign, and a successful numerical solve does not by itself establish the paper's scientific acceptance criteria.

## Main findings represented here

The code and evidence preserve both favorable and unfavorable outcomes. In particular, the manuscript reports that:

- Concentration can reduce a prescribed interface-collar error while degrading or failing to improve full-field error.
- Coverage protection is structurally useful in the reported constructions, but it has no uniformly favorable accuracy effect.
- Cross-side Robin scales give finite physical exchange on the declared circular fixtures; this does not establish general domain-decomposition convergence.
- Nonquadratic circular cases retain failures of the complete predeclared training qualification, chiefly through conservation and coverage criteria.
- In the 400-case comparison, frozen GHG has smaller aggregate full-field-error ratios than the implemented residual-adaptive control, while equal-cost superiority over Halton is not established.
- In the separate 300-case supplement, adjoint weighting improves the selected objective relative to an unweighted defect, whereas the GHG--adjoint blend is not a practical improvement over the measured Halton alternatives.
- The SPE10 calculation tests portability of the *point-selection rule* under P1 finite elements. It is not a claim of RBF-FD superiority on SPE10.

All failed cases are part of the evidence and must remain in any reproduction or reanalysis.

## Repository layout

Use the following layout. Several scripts determine the repository root from their own file location, so moving individual files can break imports or evidence discovery.

```text
.
â”œâ”€â”€ README.md
â”œâ”€â”€ LICENSE
â”œâ”€â”€ CITATION.cff
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ code/
â”‚   â”œâ”€â”€ CGF_Replay_Common_PHS_RBFFD.py
â”‚   â”œâ”€â”€ CGF_Collar_and_Component_Roles_Production_Replay.py
â”‚   â”œâ”€â”€ CGF_XM_Admissibility_and_Transfer_Qualification.py
â”‚   â”œâ”€â”€ CGF_Vug_Two_Way_Coupling_Qualification.py
â”‚   â”œâ”€â”€ CGF_Static_Interface_Production_and_Holdout.py
â”‚   â”œâ”€â”€ CGF_Smooth_Homogeneous_Nonactivation_and_Baselines.py
â”‚   â”œâ”€â”€ CGF_Frozen_Localized_Front_Activation_and_Baselines.py
â”‚   â”œâ”€â”€ CGF_SPE10_Layer68_Frozen_Halton_Primary_and_Protected_GHG_Challenger.py
â”‚   â”œâ”€â”€ CGF_Nonquadratic_Coupling_Validation.py
â”‚   â”œâ”€â”€ CGF_Adaptive_RBFFD_Frozen_Allocation_Comparison.py
â”‚   â”œâ”€â”€ CGF_Green_Adjoint_Supplement.py
â”‚   â”œâ”€â”€ CGF_Audit_Recorded_Evidence.py
â”‚   â”œâ”€â”€ CGF_Audit_Nonquadratic_Production.py
â”‚   â”œâ”€â”€ CGF_Regenerate_Paper_v3p0_Figures.py
â”‚   â”œâ”€â”€ CGF_Rebuild_Restored_Evidence.py
â”‚   â”œâ”€â”€ CGF_Audit_Green_Adjoint_Production.py
â”‚   â”œâ”€â”€ CGF_Rebuild_v3p2_Figures.py
â”‚   â”œâ”€â”€ CGF_Audit_Adaptive_Production.py
â”‚   â”œâ”€â”€ CGF_ESF_Selector_Freeze_and_SPE10_Preflight.py
â”‚   â””â”€â”€ tests/
â”‚       â””â”€â”€ verify_implementation.py
â”œâ”€â”€ evidence/
â”‚   â”œâ”€â”€ collar/
â”‚   â”œâ”€â”€ xm/
â”‚   â”œâ”€â”€ vug/
â”‚   â”œâ”€â”€ E/
â”‚   â”œâ”€â”€ F/
â”‚   â”œâ”€â”€ P/
â”‚   â”œâ”€â”€ training/
â”‚   â”œâ”€â”€ holdout/
â”‚   â”œâ”€â”€ nonquadratic/
â”‚   â””â”€â”€ adaptive_assessment/
â”œâ”€â”€ analysis/
â”‚   â””â”€â”€ green_adjoint/
â”œâ”€â”€ figures/
â””â”€â”€ data/
    â””â”€â”€ README.txt
```

`figures/` and most of `analysis/` are generated directories. Do not treat them as the sole scientific record; the immutable campaign evidence, manifests, protocols, case metrics, and saved states are the authoritative inputs.

## Software requirements

Use a current CPython installation and create an isolated environment. The retained code uses:

- Python
- NumPy
- SciPy
- pandas
- Matplotlib
- Pillow
- threadpoolctl

The adaptive driver also uses `fcntl`, so its recorded production workflow requires a Unix-like platform. The standard-library modules used by the scripts include `argparse`, `dataclasses`, `hashlib`, `json`, `pathlib`, `tempfile`, and `zipfile`.

A suitable starting environment is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy pandas matplotlib pillow threadpoolctl
```

For a formal release, record the exact package versions, Python version, operating system, BLAS/LAPACK implementation, and thread settings in `requirements.txt` or an environment lock file. The numerical campaigns should be run with the thread settings specified in their source and protocol files; timing results are machine-dependent.

## Code inventory

### Production core and experiment drivers

| File | Purpose | Manuscript results supported |
|---|---|---|
| `CGF_Replay_Common_PHS_RBFFD.py` | Shared PHS-RBF-FD solver, allocation, reconstruction, diagnostics, and interface exchange. | Dependency of several controlled campaigns. |
| `CGF_Collar_and_Component_Roles_Production_Replay.py` | Collar-width, organization, and component-removal experiments. | Figures 1--2; Table 2. |
| `CGF_XM_Admissibility_and_Transfer_Qualification.py` | Oblique/circular allocation qualification and transfer experiment. | Figure 5 and associated qualification statistics. |
| `CGF_Vug_Two_Way_Coupling_Qualification.py` | Quadratic circular-inclusion Robin-coupling experiment. | Figures 6--7; Table 7; verification input for Figure 8. |
| `CGF_Static_Interface_Production_and_Holdout.py` | Earlier static-interface training and holdout realization. | Figure 3; Tables 3--5. |
| `CGF_Smooth_Homogeneous_Nonactivation_and_Baselines.py` | Smooth homogeneous control. | Part of Figure 4; Table 6. |
| `CGF_Frozen_Localized_Front_Activation_and_Baselines.py` | Localized-layer control. | Part of Figure 4; Table 6. |
| `CGF_SPE10_Layer68_Frozen_Halton_Primary_and_Protected_GHG_Challenger.py` | SPE10 Layer 68 P1-FE/FV allocation comparison. | Figures 10--11; Table 9. |
| `CGF_Nonquadratic_Coupling_Validation.py` | 240-case nonquadratic circular-inclusion campaign. | Figure 9; Table 8. |
| `CGF_Adaptive_RBFFD_Frozen_Allocation_Comparison.py` | 400-case frozen-GHG versus adaptive comparison. | Figure 12; Table 10 after audit. |
| `CGF_Green_Adjoint_Supplement.py` | 300-case adjoint and GHG--adjoint supplement. | Figures 13--14; Tables 11--12 after audit. |

### Verification, analysis, and figure scripts

| File | Task | Required evidence |
|---|---|---|
| `CGF_Audit_Recorded_Evidence.py` | Recomputes controlled-campaign comparisons and runs four independent quadratic verification solves. | `evidence/collar`, `evidence/xm`, `evidence/vug`. |
| `CGF_Audit_Nonquadratic_Production.py` | Audits nonquadratic records; computes paired statistics, failures, convergence, and cost summaries. | Complete `evidence/nonquadratic`. |
| `CGF_Regenerate_Paper_v3p0_Figures.py` | Regenerates the retained original ten-figure set. | Original v3.0 evidence layout, including CSV and SPE10 field arrays. |
| `CGF_Rebuild_Restored_Evidence.py` | Rebuilds static-interface summaries, ablations, transmission counts, and Figure 3. | `evidence/training`, `evidence/holdout`, and `evidence/nonquadratic_campaign_metrics.csv`. |
| `CGF_Audit_Green_Adjoint_Production.py` | Verifies the complete adjoint archive, saved states, and 90 saved-adjoint checks. | Extracted 300-case adjoint production archive. |
| `CGF_Rebuild_v3p2_Figures.py` | Rebuilds manuscript Figures 12--14. | Adaptive and adjoint summary outputs. |
| `CGF_Audit_Adaptive_Production.py` | Verifies the complete 400-case adaptive archive and recomputes comparisons. | Extracted adaptive production archive. |

### Supporting provenance and tests

| File | Role |
|---|---|
| `CGF_ESF_Selector_Freeze_and_SPE10_Preflight.py` | Documents the frozen selector policy and audits localized-front evidence before the SPE10 comparison. Its embedded historical E/S summaries are not a substitute for rebuilding those campaigns from their evidence. |
| `tests/verify_implementation.py` | Checks solver/cloud agreement, deterministic budgets, polynomial reproduction, and preservation/detection of failed or changed cases. It must remain inside `code/tests/` because it resolves its drivers from its parent directory. |

The repository intentionally excludes superseded development scripts, historical figure utilities, incomplete illustrative examples, and compiled `.pyc` caches from the current-paper execution path. See `JCP_Python_Reproducibility_List.txt` in the release materials for the complete classification of the originally reviewed files.

## Evidence and data availability

The paper combines small derived tables with large campaign archives. Store large archives in a permanent data repository, such as Zenodo, rather than in Git history. Before the public release, add the DOI or stable URL here:

```text
Permanent evidence archive: [INSERT DOI OR URL]
Version used for the manuscript: [INSERT ARCHIVE VERSION/DOI]
SPE10 input source and license: [INSERT SOURCE, VERSION, AND TERMS]
```

The release must include the original manifests and protocols alongside every archive. Do not distribute only final CSV summaries when the associated audit requires individual case files or saved states.

| Campaign/archive | Why it is required | Minimum retained content |
|---|---|---|
| Controlled collar, XM, and Vug evidence | Figures 1, 2, 5, 6, 7, and 8; recorded-evidence audit. | Metrics, protocol, source snapshots, and Vug records used by the independent quadratic checks. |
| Static-interface training and holdout archives | Figure 3 and Tables 3--5. | Both campaign CSV sets and gate matrices. Run production before holdout when rerunning from scratch. |
| Smooth E and localized-front F archives | Figure 4 and Table 6. | Campaign metrics, protocol, and frozen-policy provenance. |
| Nonquadratic production archive | Figure 9 and Table 8. | `campaign_metrics.csv`, manifest, protocol, completion record, case histories, and saved records. |
| SPE10 Layer 68 archive and input data | Figures 10--11 and Table 9. | Campaign metrics, `interface_indicator_fields.npz`, `reference/reference_factor_4.npz`, input permeability data, and input provenance. |
| Adaptive production archive | Figure 12 and Table 10. | `campaign_metrics.csv`, manifest, protocol, `source/`, and all 400 case directories. |
| Green-adjoint production archive | Figures 13--14 and Tables 11--12. | `campaign.csv`, manifest, protocol, `sources/`, and all 300 case directories including `state.npz`. |

The source code alone cannot recreate an archive audit or saved-adjoint verification. Conversely, an archive without its source snapshots, manifest, and protocol cannot establish that the recorded calculation matches the declared one.

## Reproducing figures and tables from archived evidence

Start from a clean checkout and unpack the evidence archive so that the directory tree matches the layout above. The scripts write into `analysis/`, `figures/`, or a directory supplied on the command line. Preserve the original archive as read-only and write outputs outside it when an audit accepts an output directory.

### 1. Restored static-interface summaries and Figure 3

```bash
python code/CGF_Rebuild_Restored_Evidence.py
```

This reads the packaged training/holdout CSV evidence and writes summaries in `analysis/` plus the restored static-interface figure in `figures/`. It performs no PDE simulation.

### 2. Nonquadratic audit and Figure 9 inputs

```bash
python code/CGF_Audit_Nonquadratic_Production.py
```

This script reads `evidence/nonquadratic/`, verifies the manifest, and writes its analysis outputs under `analysis/`. It summarizes recorded data; it does not rerun the nonquadratic campaign.

### 3. Independent quadratic verification and Figure 8 input

```bash
python code/CGF_Audit_Recorded_Evidence.py
cp audit_replay/quadratic_patch_audit.csv analysis/quadratic_patch_audit.csv
```

The first command performs the four declared independent quadratic verification solves and writes `audit_replay/quadratic_patch_audit.csv`. The second command stages that verified output where the v3.0 figure generator expects it. Retain both copies or document the staging step in the release log.

### 4. Original retained figure set: Figures 1, 2, and 4--11

```bash
python code/CGF_Regenerate_Paper_v3p0_Figures.py
```

This requires the complete original evidence layout. In particular, it requires:

- `evidence/collar/campaign_metrics.csv`
- `evidence/E/campaign_allocation_metrics.csv`
- `evidence/F/campaign_allocation_metrics.csv`
- `evidence/xm/campaign_metrics.csv`
- `evidence/vug/branch_summary.csv` and `campaign_history.csv`
- `analysis/quadratic_patch_audit.csv`
- `evidence/nonquadratic/campaign_metrics.csv` and selected case histories
- `evidence/P/campaign_metrics.csv`
- `evidence/P/interface_indicator_fields.npz`
- `evidence/P/reference/reference_factor_4.npz`

### 5. Adaptive archive audit, Figure 12, and Table 10

```bash
python code/CGF_Audit_Adaptive_Production.py \
  /path/to/results_adaptive_comparison_production_v1 \
  analysis/adaptive_assessment
python code/CGF_Rebuild_v3p2_Figures.py
```

The audit checks the 400-case archive, its manifest, protocol, source hashes, and individual case records. `CGF_Rebuild_v3p2_Figures.py` reads `evidence/adaptive_assessment/paired_comparisons.csv`; stage the verified audit output there, or use an evidence release that already contains that file.

### 6. Green-adjoint archive audit, Figures 13--14, and Tables 11--12

```bash
python code/CGF_Audit_Green_Adjoint_Production.py \
  /path/to/results_green_adjoint_production_v1 \
  analysis/green_adjoint
python code/CGF_Rebuild_v3p2_Figures.py
```

The audit requires all 300 case directories and saved state files. It verifies manifests, source hashes, saved field RMSEs, and the adjoint/defect identities for the 90 adjoint-weighted cases. The figure builder reads `analysis/green_adjoint/recomputed_pairs.csv` and `recomputed_medians.csv`.

## Rerunning numerical campaigns

The eleven production drivers in `code/` are the source path for fresh campaigns. They are computational experiments, not quick smoke tests. Read the protocol generated by each driver and preserve the resulting source snapshot, parameter contract, random seeds, manifest, and every completed or failed case.

Important dependencies are:

```text
CGF_Replay_Common_PHS_RBFFD.py
  â”œâ”€â”€ Collar / XM / Vug / Nonquadratic drivers
  â””â”€â”€ Adaptive driver (checked dependency)
        â””â”€â”€ Green-adjoint supplement

Static-interface production -> static-interface holdout
```

The adaptive driver also uses fixtures from the nonquadratic driver. The adjoint supplement imports adaptive solver, remeshing, fixture, and diagnostic routines. Keep these files together exactly as released.

A fresh campaign is a new computational record. Compare it to the archived result, preserve any differences, and do not overwrite or relabel it as the original evidence. Exact wall times and bitwise numerical identity are not expected across different processors, BLAS libraries, operating systems, or thread settings.

## Validation expectations

Before treating a regenerated result as a reproduction:

1. Confirm that every manifest hash and protocol identifier passes.
2. Confirm that row counts, seed--budget combinations, and acceptance counts match the archived reports.
3. Include failed scientific cases in all aggregate summaries.
4. Confirm figure labels, axes, panel order, and source files against the manuscript.
5. Record the Python environment, platform, thread settings, command, input archive checksum, and output checksum.

The numerical acceptance gates are part of the result. Passing an algebraic solver criterion alone is insufficient when local, conservation, accuracy, or objective-quadrature screens fail.

## Citation

Please cite the associated article when using this code or evidence. Before publication, retain this provisional citation information in `CITATION.cff` and replace the bracketed fields when the DOI is assigned:

```text
Ortiz-Tapia, Arturo, and DÃ­az-Viera, MartÃ­n Alberto.
â€œBenefits and limits of physics-informed point allocation for elliptic
interface problems: Controlled RBF--FD, adjoint, and Robin-coupling
comparisons.â€ Journal of Computational Physics, [year], [DOI].
```

If the code and data obtain a separate Zenodo DOI, cite that archived release as well as the article.

## License and commercial use

Unless a file states otherwise, this repository is distributed under the **Creative Commons Attribution--NonCommercial 4.0 International** license (CC BY-NC 4.0):

<https://creativecommons.org/licenses/by-nc/4.0/>

You may share and adapt the materials for noncommercial purposes with appropriate attribution and an indication of changes. Commercial use requires prior written permission from the copyright holders and a separate commercial license. CC BY-NC 4.0 restricts commercial use; it does not itself set a fee or automatically create a payment obligation.

Before public release, confirm that every included external dataset, especially the SPE10 input data, permits redistribution under the terms selected for this repository. Preserve third-party notices and licenses with their files.

## Contact

**Corresponding author:** Arturo Ortiz-Tapia  
<egl.arturo.ortizta@unadmexico.mx>

For reproducibility questions, please open a GitHub issue with the repository release tag, operating system, Python and package versions, exact command, input archive checksum, and the complete error output.

## Release checklist

Before making the repository public:

- [ ] Add the 20 retained Python sources listed above, with the six recovered postprocessing scripts included.
- [ ] Add `LICENSE` containing the complete CC BY-NC 4.0 legal code, or revise this README if a different license is selected for software.
- [ ] Add `CITATION.cff` and replace placeholder DOI/year fields after publication or archival deposit.
- [ ] Add an environment file with tested package versions.
- [ ] Upload the complete evidence archives to a persistent repository and insert their DOI/URL above.
- [ ] Include archive manifests, protocols, source snapshots, and case-level records, not only summary tables.
- [ ] Verify that the two SPE10 field arrays required by the figure generator are present.
- [ ] Run the audit and figure-generation commands from a clean checkout and record the results.
- [ ] Check third-party data licenses and remove any files that cannot be redistributed.
- [ ] Create a tagged GitHub release that matches the archived data release.
