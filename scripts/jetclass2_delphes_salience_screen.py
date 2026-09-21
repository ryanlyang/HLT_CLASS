"""Create, submit, or run the JetClass2 salience U100 screen."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.jetclass2_delphes.salience_screen import create_screen, run_task, submit_screen

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="mode",required=True)
    c=sub.add_parser("create"); c.add_argument("--bottleneck-root",type=Path,required=True); c.add_argument("--candidate-root",type=Path,action="append",required=True); c.add_argument("--resource-template",type=Path,required=True); c.add_argument("--data-root",type=Path,required=True); c.add_argument("--output-root",type=Path,required=True); c.add_argument("--source-commit",required=True); c.add_argument("--screen-execution-site",choices=("sporc_a100","sporc_a100_debug"),default="sporc_a100"); c.add_argument("--debug-walltime-minutes",type=int,default=480)
    r=sub.add_parser("run"); r.add_argument("--spec",type=Path,required=True); r.add_argument("--task",required=True); r.add_argument("--attempt",required=True)
    s=sub.add_parser("submit"); s.add_argument("--spec",type=Path,required=True); s.add_argument("--execute",action="store_true"); s.add_argument("--authorization-phrase")
    z=sub.add_parser("results"); z.add_argument("--spec",type=Path,required=True)
    a=p.parse_args()
    if a.mode=="create": result=create_screen(bottleneck_root=a.bottleneck_root,candidate_roots=a.candidate_root,resource_template=a.resource_template,data_root=a.data_root,output_root=a.output_root,project=ROOT,source_commit=a.source_commit,screen_execution_site_name=a.screen_execution_site,debug_walltime_minutes=a.debug_walltime_minutes)
    elif a.mode=="run": result=run_task(load_json(a.spec),a.task,attempt=a.attempt)
    elif a.mode=="submit": result=submit_screen(load_json(a.spec),execute=a.execute,authorization_phrase=a.authorization_phrase)
    else:
        spec=load_json(a.spec); root=Path(spec["screen_root"])
        print(f"{'candidate':<32} {'state':<9} {'pick':>8} {'accuracy':>10} {'AUC':>10} {'R50':>10}")
        for name in ["BOTTLENECK_CONTEXT", *spec["candidate_registry"]]:
            path=root/"tasks"/f"fit_{name}.json"
            if not path.is_file(): print(f"{name:<32} {'PENDING':<9}"); continue
            fit=load_json(path)["result"]; m=fit["selection"]; r50=m.get("macro_r50")
            print(f"{name:<32} {'COMPLETE':<9} {fit['training']['selected_pass']:>3}/{fit['training']['passes']:<4} {m['accuracy']:>10.6f} {m['macro_ovr_auc']:>10.6f} {('n/a' if r50 is None else f'{r50:.1f}'):>10}")
        lock=root/"selection_lock.json"
        print("Winner:", load_json(lock)["winner"] if lock.is_file() else "PENDING")
        return
    print(result["content_hash"])
if __name__=="__main__": main()
