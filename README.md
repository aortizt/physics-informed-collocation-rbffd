# Physics-informed collocation for elliptic interface problems

Arturo Ortiz-Tapia and Martín Alberto Díaz-Viera

Code and evidence supporting *Benefits and limits of physics-informed point allocation for elliptic interface problems: Controlled allocation, equation-enforcement, and interface-coupling comparisons*.

## Scope

The study reports controlled successes and failures of Green–Halton–Gradient allocation, adaptive and adjoint alternatives, Robin coupling, and a separate fixed-node equation-enforcement experiment. Conclusions are specific to the tested implementation, comparator, metric and configuration. The paper does not claim a universally superior interface solver.

## Current repository layout

The earlier production drivers remain in the repository root. This release adds `fixed_node_study/`; do not relocate earlier drivers as part of this update.

- Root `CGF_*.py`: earlier campaign drivers and common engine.
- `JCP_Recovered_Postprocessing.zip`: six earlier postprocessing sources with their required layout. `README.txt` describes these recovered sources; it is not the general repository README.
- Existing result ZIPs at the root: retained earlier materials, not replaced by the new study.
- `fixed_node_study/campaign_sources/`: new foundation, recovery and frozen-validation implementations plus required reference evidence.
- `fixed_node_study/regenerate_new_results.py`: Spyder-compatible postprocessing of the supplied results; no PDE runs.
- `fixed_node_study/data/`, `tables/`, `figures/`: case-level summaries, manuscript tables and all figure assets.
- `fixed_node_study/CGF_Paper_I_MATCOM_v3p6.tex` and `.pdf`: author-identified transfer-preparation manuscript.

The prior root drivers are:

- `CGF_Adaptive_RBFFD_Frozen_Allocation_Comparison.py`
- `CGF_Audit_Recorded_Evidence.py`
- `CGF_Collar_and_Component_Roles_Production_Replay.py`
- `CGF_ESF_Selector_Freeze_and_SPE10_Preflight.py`
- `CGF_Frozen_Localized_Front_Activation_and_Baselines.py`
- `CGF_Green_Adjoint_Supplement.py`
- `CGF_Nonquadratic_Coupling_Validation.py`
- `CGF_Replay_Common_PHS_RBFFD.py`
- `CGF_SPE10_Layer68_Frozen_Halton_Primary_and_Protected_GHG_Challenger.py`
- `CGF_Smooth_Homogeneous_Nonactivation_and_Baselines.py`
- `CGF_Static_Interface_Production_and_Holdout.py`
- `CGF_Vug_Two_Way_Coupling_Qualification.py`
- `CGF_XM_Admissibility_and_Transfer_Qualification.py`

## Reproduce the fixed-node results

Read `fixed_node_study/README.md`. Place the three result archives in `fixed_node_study/analysis_inputs/`, then press F5 on `regenerate_new_results.py` in Spyder. It regenerates the new tables and Figures 15–16. Complete package folders are required to rerun the numerical campaigns; do not copy only the numbered entry-point scripts.

Earlier drivers and recovered postprocessing have their own input layouts and requirements. Their scientific data remain in the earlier deposit. Keeping failed cases is essential to reproducing the reported conclusions.

## Numerical findings

Each frozen LS oversampling candidate qualifies on all three fresh highest-budget vertical cases and all six lower-contrast oblique cases, but fails on all six harder oblique cases. The harder failures concern independent conormal continuity even though other physical and solver criteria pass. Changing equation sampling on fixed nodes is distinct from improving allocation. Timings from this diagnostic implementation do not establish production efficiency.

## Data and software archives

- Data collection, all versions: https://doi.org/10.5281/zenodo.22850116
- Earlier data version: https://doi.org/10.5281/zenodo.22850117
- Software collection, all versions: https://doi.org/10.5281/zenodo.22730371
- Earlier software release v1.0.0: https://doi.org/10.5281/zenodo.22850313

The three new result archives must be included in the corresponding new data version before the transfer. Until that version is published, the older data DOI does not contain them. Record the new version-specific DOI and release tag in the final manuscript and release notes. The new release is planned as v1.1.0; a README edit alone does not create an archived software release.

## Software and external data

The new study uses NumPy, SciPy and threadpoolctl; its postprocessing uses Matplotlib. See `fixed_node_study/requirements_fixed_node.txt`. Earlier workflows may also use pandas and Pillow. Follow each campaign's thread settings; exact environments are recorded in the supplied results.

SPE10 is third-party input data. Its provenance is described in the existing paper and data archive; its original terms remain applicable. This update adds no new SPE10 files.

## License

The repository's existing notice is `License CC BY-NC 4.0`, and applies as stated there. This update does not change that license. Third-party materials retain their own terms. The repository and Zenodo license metadata should be reconciled by the authors before publishing the new release.
