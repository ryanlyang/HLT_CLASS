# CMS_FULLSIM_FEATURE_LADDER/v1

Authority: [scientific plan](../plans/CMS_FULLSIM_SHARED_FEATURE_LADDER_200K_PLAN.md).

New namespace, never relabel older learned-fusion or proxy campaign artifacts.
SPEC, FOUNDATION, ASSIGNMENT, PREFLIGHT, TASK, FIT, TRAINING, BANK, RESULTS and
PLAN artifacts carry canonical SHA256, exact parent references, source-commit
lineage through the pinned campaign, and final_test_accessed=false.
Physical byte references are checked before reuse. Original source roles and
all selected identities are preserved; final-test branch access has no API.
The default arm list is [SHARED17]; CMS21 must be explicitly registered.

PREPARE -> source-local MATCH tasks -> FOUNDATION -> per-arm PREFLIGHT forms
the gate. Science is a separate reviewed DAG with per-arm isolated teachers.
Scheduler completion is not enough: foundation and acceptance receipts must
authenticate before science submission and every worker. Submission requires
a clean exact pushed checkout, the saved plan hash and explicit authorization.
Inherited Slurm/SBATCH settings are removed; jobs use absolute helper paths,
PYTHONNOUSERSITE=1 and CONDA_PREFIX/lib first. No automatic cancellation.

Training reports label the selected representation and native 15-class schema.
Recovery denominators are arm-specific. Dataset source is genuine CMS FullSim;
the selected common representation does not claim identical detector response,
class mixture, or feature information to public JetClass2.
