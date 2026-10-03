"""Closed registry for the CMS feature-interface comparison."""
from copy import deepcopy
from hlt_classification.data.cache_contracts import with_content_hash, validate_content_hash
from hlt_classification.jetclass2_delphes.campaign import recipe
from hlt_classification.cms_proxy_ladder.campaign import paired_seed
from hlt_classification.scouting.schema import CLASS_NAMES

FAMILY = "CMS_FULLSIM_FEATURE_LADDER"
BUDGETS = {"train": 200000, "validation": 50000}
ARMS = ("SHARED17", "CMS21")
COARSE = ("U050", "U100", "D066", "D033", "D000")
AUTHORIZATION = "AUTHORIZE CMS FULLSIM FEATURE LADDER EXACT PLAN"
CAPACITY = 512


def artifact(kind, *, parents=None, **fields):
    return with_content_hash(dict(contract=f"{FAMILY}_{kind}/v1", schema_version=1,
        parents=parents or {}, final_test_accessed=False, **fields))


def validate(value, kind, *, parents=None):
    digest = validate_content_hash(value, expected_contract=f"{FAMILY}_{kind}/v1", expected_schema_version=1)
    if value.get("final_test_accessed") is not False or (parents is not None and value["parents"] != parents):
        raise ValueError("CMS feature artifact access/lineage differs")
    return digest


def nodes(arm):
    if arm not in ARMS:
        raise ValueError("Unknown CMS feature arm")
    rows = [dict(name=n, coordinate=c, teacher=None, branch="CONTROL")
            for n, c in (("M0HLT", "D000"), ("OFFLINE", "OFFLINE"), ("U000", "U000"))]
    rows.append(dict(name="DIRECT_D000", coordinate="D000", teacher="U000", branch="DIRECT"))
    parent = "U000"
    for coordinate in COARSE:
        name = "COARSE_" + coordinate
        rows.append(dict(name=name, coordinate=coordinate, teacher=parent, branch="COARSE"))
        parent = name
    for row in rows:
        row.update(node_id=arm + "_" + row["name"], arm=arm,
            initialization_seed=paired_seed(row["coordinate"], "initialization"),
            sampler_seed=paired_seed(row["coordinate"], "sampler"))
    return rows


def registry(arms):
    if list(arms) not in (["SHARED17"], ["CMS21"], list(ARMS)):
        raise ValueError("Register SHARED17, CMS21, or both in canonical order")
    return dict(arms=list(arms), budgets=deepcopy(BUDGETS), capacity=CAPACITY,
        classes=list(CLASS_NAMES), training=recipe(), trim=False,
        matcher="SALIENCE_PT_LINEAR", support="persistent_hlt_shell_offline_tail",
        source="genuine_CMS_FullSim", cms_length_to_mm=10., lost_tracks="retained",
        nodes=[n for arm in arms for n in nodes(arm)])


def science_tasks(arms):
    tasks = []
    for arm in arms:
        for row in nodes(arm):
            task = "train_" + row["node_id"]
            dependencies = [] if row["teacher"] is None else ["reduce_" + arm + "_" + row["teacher"]]
            tasks.append(dict(task_id=task, kind="train", arm=arm, node=row["node_id"], dependencies=dependencies))
            if row["name"] in ("U000", "COARSE_U050", "COARSE_U100", "COARSE_D066", "COARSE_D033"):
                tasks.append(dict(task_id="reduce_" + row["node_id"], kind="reduce", arm=arm,
                                  node=row["node_id"], dependencies=[task]))
        dependencies = [t["task_id"] for t in tasks if t["arm"] == arm]
        tasks.append(dict(task_id="aggregate_" + arm, kind="aggregate", arm=arm, node=None, dependencies=dependencies))
        tasks.append(dict(task_id="complete_" + arm, kind="complete", arm=arm, node=None, dependencies=["aggregate_" + arm]))
    return tasks
