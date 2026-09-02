#!/usr/bin/env python3
"""Print completed Strategy-B rows with M0CE60-to-U000 recovery."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
def r50(metrics):return math.exp(float(metrics["macro_mean_log_qcd_rejection_at_50pct_signal"]))
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--campaign-root",type=Path,required=True);a=p.parse_args();root=a.campaign_root.resolve();aggregate=json.loads((root/"reports/validation_aggregate.json").read_text());controls={x["model_id"]:x["metrics"] for x in aggregate["control_rows"]};base=controls["M0CE60"];oracle=controls["U000"]
 def recovery(value,lo,hi):return 100*(value-lo)/(hi-lo) if hi!=lo else float("nan")
 print(f"Campaign: {root}\nFinal test accessed: {aggregate['final_test_accessed']}\n")
 print(f"{'model':<42} {'kind':<24} {'pick':>7} {'done':>5} {'hours':>7} {'accuracy':>10} {'AUC':>10} {'R50':>10} {'AUC rec.':>10} {'R50 rec.':>10} {'RAM GiB':>9} {'GPU GiB':>9} {'disk MiB':>10}")
 rows=[{"model_id":"M0CE60","kind":"baseline","metrics":base},{"model_id":"U000","kind":"offline oracle","metrics":oracle},*aggregate["model_rows"]]
 for row in rows:
  m=row["metrics"];auc=float(m["macro_ovr_auc"]);rejection=r50(m);training=row.get("training",{});stored=row.get("recovery_m0ce60_to_u000",{});auc_recovery=stored.get("auc_pct",recovery(auc,float(base["macro_ovr_auc"]),float(oracle["macro_ovr_auc"])));r50_recovery=stored.get("macro_r50_linear_pct",recovery(rejection,r50(base),r50(oracle)));pick=str(training.get("selected_pass","—"));done=str(training.get("completed_passes","—"));hours=float(training.get("runtime_seconds",0))/3600;ram=float(training.get("peak_cpu_rss_bytes",0))/1024**3;gpu=float(training.get("peak_gpu_memory_bytes",0))/1024**3;disk=float(training.get("durable_output_bytes",0))/1024**2;print(f"{row['model_id']:<42} {row['kind']:<24} {pick:>7} {done:>5} {hours:>7.2f} {float(m['accuracy']):>10.6f} {auc:>10.6f} {rejection:>10.1f} {auc_recovery:>+9.1f}% {r50_recovery:>+9.1f}% {ram:>9.1f} {gpu:>9.1f} {disk:>10.1f}")
 print("\nREQUIRED CAUSAL COMPARISONS")
 print(f"{'left minus right':<88} {'dAUC':>10} {'AUC 95% CI':>25} {'dR50':>10}")
 for row in aggregate["required_causal_comparisons"]:
  boot=row["paired_macro_auc_bootstrap"]
  name=f"{row['left']} - {row['right']}"
  interval=f"[{float(boot['lower_95']):+.6f}, {float(boot['upper_95']):+.6f}]"
  print(f"{name:<88} {float(row['delta_macro_ovr_auc']):>+10.6f} {interval:>25} {float(row['delta_macro_r50_linear']):>+10.1f}")
 print("\nADJACENT EXTRACTED-CARRIER TRANSITIONS")
 for row in aggregate["adjacent_carrier_comparisons"]:
  boot=row["paired_macro_auc_bootstrap"]
  print(f"{row['left']} - {row['right']}: dAUC={float(row['delta_macro_ovr_auc']):+.6f}  95% CI=[{float(boot['lower_95']):+.6f}, {float(boot['upper_95']):+.6f}]  dR50={float(row['delta_macro_r50_linear']):+.1f}")
 print("\nLEARNED CARRIER MINUS MATCHED DIRECT-KD MODEL")
 for row in aggregate["learned_carrier_minus_direct_comparisons"]:
  boot=row["paired_macro_auc_bootstrap"]
  print(f"{row['left']} - {row['right']}: dAUC={float(row['delta_macro_ovr_auc']):+.6f}  95% CI=[{float(boot['lower_95']):+.6f}, {float(boot['upper_95']):+.6f}]  dR50={float(row['delta_macro_r50_linear']):+.1f}")
 print("\nPER-RUNG CONTEXT / WITHDRAWAL DECOMPOSITION")
 print(f"{'rung':<6} {'context dAUC':>14} {'withdraw dAUC':>15} {'AUC recovered':>15} {'context dR50':>14} {'withdraw dR50':>15} {'R50 recovered':>15}")
 for row in aggregate["rung_withdrawal_decomposition"]:
  auc_pct=row["auc_context_gain_recovered_pct"]
  r50_pct=row["r50_context_gain_recovered_pct"]
  auc_text="n/a" if auc_pct is None else f"{float(auc_pct):+.1f}%"
  r50_text="n/a" if r50_pct is None else f"{float(r50_pct):+.1f}%"
  print(f"{row['coordinate']:<6} {float(row['auc_context_gain_before_withdrawal']):>+14.6f} {float(row['auc_gain_recovered_by_withdrawal']):>+15.6f} {auc_text:>15} {float(row['r50_context_gain_before_withdrawal']):>+14.1f} {float(row['r50_gain_recovered_by_withdrawal']):>+15.1f} {r50_text:>15}")
 print("\nSELECTED-CHECKPOINT ALPHA VALIDATION CURVES")
 for row in aggregate["model_rows"]:
  curve=row.get("diagnostics",{}).get("alpha_validation_curve")
  if not curve:continue
  values="  ".join(f"a={float(point['alpha']):.2f}:AUC={float(point['metrics']['macro_ovr_auc']):.6f},R50={r50(point['metrics']):.1f}" for point in curve)
  print(f"{row['model_id']}: {values}")
 return 0
if __name__=="__main__":raise SystemExit(main())
