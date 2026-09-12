RECOVERED JCP PAPER I POSTPROCESSING SOURCES
=========================================

These are the six existing source files previously marked as missing from
GitHub_Collocation.zip. They were recovered from the retained JCP preparation
materials and checked against the corresponding v3.3 review copies. All six
match those review copies exactly. They have not been reverse engineered,
rewritten, or changed to fit the published numbers.

CONTENTS
--------

code/CGF_Audit_Nonquadratic_Production.py
code/CGF_Regenerate_Paper_v3p0_Figures.py
code/CGF_Rebuild_Restored_Evidence.py
code/CGF_Audit_Green_Adjoint_Production.py
code/CGF_Rebuild_v3p2_Figures.py
code/CGF_Audit_Adaptive_Production.py

WHY INCLUDE THEM?
-----------------

GitHub itself does not require these particular scripts. They are part of
the current paper's reproducibility set because they connect the saved
experiments to the analyses and figures reported in the manuscript.

The three Audit scripts recompute numerical summaries, comparisons, or
checks from recorded results. The three Regenerate/Rebuild scripts produce
the final figure set; Rebuild_Restored_Evidence also recomputes static
summaries and nonquadratic failure intersections.

The main simulation drivers can run without these six scripts. A repository
containing only those drivers would nevertheless omit this existing route
from their output to the paper's numerical analyses and figures.

PUTTING THEM IN YOUR REPOSITORY
------------------------------

After extracting this package, copy its code/ directory into the repository
root, preserving the six filenames and their contents. If code/ already
exists, merge the directories and preserve any conflicting version for
comparison before replacing it. These copies do not need source adaptation
merely to be included in GitHub.

Four scripts infer the evidence root from their position one directory
below it. Thus repository_root/code/script.py resolves repository_root as
the package root. Running them in a flat repository root would instead
resolve its parent, which is the wrong location for the intended layout.

THE OLD v3.0 ZIP
---------------

CGF_Paper_I_v3p0_JCP_Revision.zip is an earlier packaging artifact, not a
runtime dependency opened by any of these six scripts. Its exact filename
and container are not essential. An equivalent collection of the original
inputs in the expected layout can serve the same purpose.

That old ZIP was not recovered in this step. Only the six sources listed
above and this guide are included in this recovery download.

INPUTS STILL NEEDED
-------------------

1. CGF_Audit_Nonquadratic_Production.py
   Expects the complete nonquadratic campaign under evidence/nonquadratic/,
   including campaign_metrics.csv, scientific_manifest_sha256.csv,
   protocol.json, production_complete.json, and the files the manifest names.
   Writes analysis/. Check its reported audit failures; script completion
   by itself does not establish that the input archive was complete.

2. CGF_Regenerate_Paper_v3p0_Figures.py
   Expects evidence/collar/, evidence/xm/, evidence/vug/, evidence/E/,
   evidence/F/, evidence/nonquadratic/, and evidence/P/ below the root.
   Also requires analysis/quadratic_patch_audit.csv.
   It rebuilds ten retained figures: manuscript Figures 1, 2, 4-11.

   Three figure inputs absent from the retained v3.3 review folder are:
     analysis/quadratic_patch_audit.csv
     evidence/P/interface_indicator_fields.npz
     evidence/P/reference/reference_factor_4.npz

   The quadratic CSV can be recomputed with the already supplied
   CGF_Audit_Recorded_Evidence.py, using its original controlled evidence
   and source layout. That audit performs the four verification solves
   and writes the CSV to audit_replay/. Its verified output must then be
   staged at the analysis/ path expected by this figure generator.

   The two SPE10 arrays should first be sought inside the actual
   spe10_layer68_frozen_campaign.zip or its extracted campaign folder.
   If absent there too, the SPE10 source and recorded input/protocol can
   support a new computation. Newly computed output must be identified as
   such and compared with the recorded results; it is not recovered data.
   Final figure PDFs cannot supply the original numerical arrays reliably.

   The retained review folder contains the other major scalar inputs and
   240 nonquadratic histories. Their presence does not establish that the
   complete production archives, states, and environment have been recovered.

3. CGF_Rebuild_Restored_Evidence.py
   Expects these files below the root:
     evidence/training/campaign_allocation_metrics.csv
     evidence/training/campaign_ablation_metrics.csv
     evidence/training/campaign_gate_matrix.csv
     evidence/holdout/campaign_allocation_metrics.csv
     evidence/holdout/campaign_ablation_metrics.csv
     evidence/holdout/campaign_gate_matrix.csv
     evidence/nonquadratic_campaign_metrics.csv
   These inputs are present in the retained v3.3 review folder.
   Rebuilds Figure 3 and the associated summaries.

4. CGF_Audit_Green_Adjoint_Production.py
   Takes two arguments: the extracted full adjoint RESULTS directory and
   a separate OUTPUT directory. Reads campaign.csv, protocol.json, sources/,
   the manifest, and the complete cases/ records including state.npz.
   The large results_green_adjoint_production_v1.zip supplies that evidence.
   This recovery package does not contain those large saved states.

5. CGF_Rebuild_v3p2_Figures.py
   Expects these files below the root:
     evidence/adaptive_assessment/paired_comparisons.csv
     analysis/green_adjoint/recomputed_pairs.csv
     analysis/green_adjoint/recomputed_medians.csv
   These inputs are present in the retained v3.3 review folder.
   Rebuilds Figures 12, 13, and 14.

6. CGF_Audit_Adaptive_Production.py
   Takes two arguments: the extracted full adaptive RESULTS directory and
   a separate OUTPUT directory. Reads campaign_metrics.csv, protocol.json,
   the manifest, source/, and the per-case records and recorded payloads.
   The large results_adaptive_comparison_production_v1.zip supplies that
   evidence. Summary CSVs alone do not satisfy its full-archive checks.

The retained review places earlier experiment material under evidence/legacy/.
The v3.0 scripts expect it directly under evidence/. This directory mapping
must be restored explicitly before execution; changing the working directory
alone does not change paths derived from __file__.

PYTHON DEPENDENCIES
-------------------

Across these six scripts: NumPy, SciPy, pandas, Matplotlib, and Pillow.
Pillow is imported as PIL by the v3.0 figure generator. It was omitted from
the package summary in the preceding source-list review and is required.
Use the recorded campaign/reproduction environment when it is available;
this recovery did not establish a new tested environment or version lock.

If code truly had been lost, a new implementation could be written from
the documented method and verified against original cases. The repository
would need to identify it as a reconstruction. That is unnecessary for
these six files because the existing source copies survived.

VALIDATION OF THIS RECOVERY
---------------------------

[x] All six source files recovered and matched against retained review copies.
[x] All six source files parse as Python.
[x] Source contents preserved exactly in the downloadable ZIP.
[x] ASCII guide supplied with source roles, placement, and input limitations.
[ ] Complete evidence/layout assembled in the user's GitHub repository.
[ ] End-to-end analyses and figure reproduction checked in that repository.

No simulation, numerical audit, or figure-generation script was executed
during this recovery. No data were invented and no live GitHub change was made.
