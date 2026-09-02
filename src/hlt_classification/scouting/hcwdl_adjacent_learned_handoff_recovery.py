"""Restart-zero source-pinned recovery for learned fusion handoff."""
from __future__ import annotations
from pathlib import Path
import re,shutil
from typing import Any,Mapping
from hlt_classification.data.cache_contracts import load_json,validate_content_hash,write_immutable_json
from .hcwdl_adjacent_learned_handoff_campaign import JOB_PREFIX,RECOVERY_SUBMISSION_PHRASE,RESOURCES,validate_campaign
from .hcwdl_adjacent_learned_handoff_contracts import RECOVERY_PLAN_CONTRACT,RECOVERY_SPEC_CONTRACT,artifact,validate_artifact
from .hcwdl_adjacent_learned_handoff_workflow import task_outputs
from .hcwdl_mhpe_tri60_campaign import ACCOUNT,PARTITION
from .hcwdl_recovery import MONITOR_CONTRACT,resume_tasks,validate_submission_ledger
def create_recovery(*,campaign_spec:str|Path,submission_ledger:str|Path,monitor_report:str|Path,recovery_root:str|Path,project_dir:str|Path,source_commit:str,publish:bool=True):
 if re.fullmatch(r"[0-9a-f]{40}",source_commit) is None:raise ValueError("learned-handoff recovery source commit differs")
 spec_path=Path(campaign_spec).resolve();spec=load_json(spec_path);validate_campaign(spec);ledger_path=Path(submission_ledger).resolve();ledger=load_json(ledger_path);ledger_hash=validate_submission_ledger(ledger);monitor_path=Path(monitor_report).resolve();monitor=load_json(monitor_path);monitor_hash=validate_content_hash(monitor,expected_contract=MONITOR_CONTRACT,expected_schema_version=1)
 if ledger.get("campaign_spec_sha256")!=spec["content_hash"] or monitor.get("submission_ledger_sha256")!=ledger_hash or any(row.get("disposition")=="active_or_unknown" for row in monitor["rows"]):raise ValueError("learned-handoff recovery subject is not terminal")
 task_ids=set(ledger["jobs"]);available={row["task_id"] for row in spec["tasks"]}
 if not task_ids or not task_ids<=available:raise ValueError("learned-handoff recovery task coverage differs")
 graph={row["task_id"]:tuple(x for x in row["dependencies"] if x in task_ids) for row in spec["tasks"] if row["task_id"] in task_ids};retry=resume_tasks(monitor,dependency_graph=graph)
 if not retry:raise ValueError("learned-handoff recovery has no failed closure")
 root=Path(recovery_root).resolve();project=Path(project_dir).resolve()
 if publish and root.exists():raise FileExistsError("learned-handoff recovery root exists")
 recovery=artifact({"subject_campaign_spec_path":str(spec_path),"subject_submission_ledger_path":str(ledger_path),"monitor_report_path":str(monitor_path),"recovery_root":str(root),"project_dir":str(project),"source_commit":source_commit,"parents":{"campaign_spec":spec["content_hash"],"submission_ledger":ledger_hash,"monitor":monitor_hash},"retry_tasks":list(retry),"restart_from_update_zero":True,"scientific_semantics_unchanged":True,"final_test_accessed":False},contract=RECOVERY_SPEC_CONTRACT)
 registry={row["task_id"]:row for row in spec["tasks"]};commands=[]
 for task_id in retry:
  task=registry[task_id];resource=RESOURCES[task["resource"]];dependencies=[x for x in task["dependencies"] if x in retry];command=["sbatch","--parsable",f"--account={ACCOUNT}",f"--partition={PARTITION}","--nodes=1","--ntasks=1",f"--cpus-per-task={resource.cpus}",f"--mem={resource.memory}",f"--time={resource.walltime}",f"--job-name={JOB_PREFIX}r_{task_id}"]
  if resource.gpu:command.append(f"--gres={resource.gpu}")
  if dependencies:command.append("--dependency=afterok:"+":".join(f"${{JOB_{x}}}" for x in dependencies))
  command.extend(("--export=ALL,"+f"PROJECT_DIR={project},HCWDL_LFH_RECOVERY_SPEC={root/'recovery_spec.json'},HCWDL_LFH_TASK={task_id}",str(project/"sbatch/run_hcwdl_adjacent_learned_handoff_recovery_task.sh")));commands.append({"task_id":task_id,"dependencies":dependencies,"command":command})
 plan=artifact({"recovery_spec_sha256":recovery["content_hash"],"commands":commands,"restart_from_update_zero":True,"final_test_accessed":False},contract=RECOVERY_PLAN_CONTRACT)
 if publish:
  root.mkdir(parents=True,exist_ok=False);write_immutable_json(root/"recovery_spec.json",recovery);write_immutable_json(root/"command_plan.json",plan)
 return recovery
def validate_recovery(value:Mapping[str,Any])->str:
 digest=validate_artifact(value,contract=RECOVERY_SPEC_CONTRACT)
 if value.get("restart_from_update_zero") is not True or value.get("scientific_semantics_unchanged") is not True:
  raise ValueError("learned-handoff recovery semantics differ")
 return digest
def clean_incomplete_task_outputs(spec,task_id):
 root=Path(spec["campaign_root"]).resolve();targets=[]
 task={row["task_id"]:row for row in spec["tasks"]}[task_id]
 if task["kind"]=="train":targets=[Path(spec["campaign_root"])/"training"/task["node_id"]]
 elif task["kind"]=="extract":targets=[Path(spec["campaign_root"])/"deployable"/task["distribution_id"],Path(spec["campaign_root"])/"reports/diagnostics"/f"{task['node_id']}.json"]
 elif task["kind"] in {"source_reducer","model_reducer","extracted_reducer"}:
  if task["kind"]=="source_reducer":distribution=task["distribution_id"]
  elif task["kind"]=="extracted_reducer":distribution=task["distribution_id"]
  else:
   from .hcwdl_adjacent_learned_handoff_workflow import _node_distribution
   distribution=_node_distribution(task["node_id"])
  targets=[Path(spec["campaign_root"])/"probabilities"/distribution,Path(spec["campaign_root"])/"reports/stages"/f"{distribution}.json",Path(spec["campaign_root"])/"reports/diagnostics"/f"{task.get('node_id','')}.json",Path(spec["campaign_root"])/"reports/diagnostics"/f"{task.get('node_id','')}_alpha_zero_V_report.npz"]
 else:targets=task_outputs(spec,task_id)
 for path in targets:
  resolved=Path(path).resolve()
  if root not in resolved.parents and resolved!=root:raise PermissionError("learned-handoff cleanup escaped campaign root")
  if resolved.is_dir():shutil.rmtree(resolved)
  elif resolved.exists():resolved.unlink()
__all__=["RECOVERY_SUBMISSION_PHRASE","clean_incomplete_task_outputs","create_recovery","validate_recovery"]
