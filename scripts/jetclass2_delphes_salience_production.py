"""Create, run, submit, monitor, or recover the JetClass2 salience campaign."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.jetclass2_delphes.salience_production import create_campaign,monitor,prepare_recovery,result_rows,run_task,submit_campaign
def main():
 p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="mode",required=True)
 c=sub.add_parser("create"); c.add_argument("--screen-spec",type=Path,required=True); c.add_argument("--data-root",type=Path,required=True); c.add_argument("--output-root",type=Path,required=True); c.add_argument("--source-commit",required=True)
 for mode in ("run","submit","monitor","recover","results"):
  q=sub.add_parser(mode); q.add_argument("--spec",type=Path,required=True)
  if mode=="run": q.add_argument("--task",required=True); q.add_argument("--attempt",required=True)
  if mode=="submit": q.add_argument("--bookkeeping-root",type=Path); q.add_argument("--execute",action="store_true"); q.add_argument("--authorization-phrase")
  if mode in {"monitor","recover"}: q.add_argument("--ledger",type=Path,required=True)
  if mode=="recover": q.add_argument("--output-root",type=Path,required=True)
 a=p.parse_args()
 if a.mode=="create": result=create_campaign(screen_spec_path=a.screen_spec,data_root=a.data_root,campaign_root=a.output_root,project=ROOT,source_commit=a.source_commit)
 else:
  spec=load_json(a.spec)
  if a.mode=="run": result=run_task(spec,a.task,attempt=a.attempt)
  elif a.mode=="submit": result=submit_campaign(spec,bookkeeping_root=a.bookkeeping_root or Path(spec["campaign_root"]),execute=a.execute,authorization_phrase=a.authorization_phrase)
  elif a.mode=="monitor":
   result=monitor(spec,load_json(a.ledger))
   for row in result["rows"]: print(f"{row['job_id']:>12} {row['task_id']:<62} {row['state']:<18} outputs={row['outputs_complete']}")
  elif a.mode=="recover": result=prepare_recovery(spec,load_json(a.ledger),output_root=a.output_root)
  else:
   print(f"{'node':<52} {'state':<9} {'pick':>8} {'accuracy':>10} {'AUC':>10} {'R50':>10} {'AUC rec':>10} {'R50 rec':>10}")
   for row in result_rows(spec):
    m=row['validation'] or {}; rec=row['recovery'] or {}; fmt=lambda x,d=6:'n/a' if x is None else f'{x:.{d}f}'; pick='n/a' if row['passes'] is None else f"{row['selected_pass']}/{row['passes']}"
    print(f"{row['node_id']:<52} {row['state']:<9} {pick:>8} {fmt(m.get('accuracy')):>10} {fmt(m.get('macro_ovr_auc')):>10} {fmt(m.get('macro_r50'),1):>10} {fmt(rec.get('macro_ovr_auc'),1):>10} {fmt(rec.get('macro_r50'),1):>10}")
   return
 print(result["content_hash"])
if __name__=="__main__": main()
