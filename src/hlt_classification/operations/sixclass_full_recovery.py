"""Full-validation DE000 epoch comparison; fixed historical/new anchors.

Loaded against the unmodified lower-LR execution checkout. See the versioned
analysis plan: no training or checkpoint reselection is performed.
"""
import os
from pathlib import Path
import time

import numpy as np

from hlt_classification.operations import sixclass_hbb_reselection as hbb
from hlt_classification.data.cache_contracts import (
    load_json, with_content_hash, validate_content_hash, write_immutable_json,
)
from hlt_classification.cms_proxy_ladder.contracts import validate_file_ref

check = hbb.check
POLICY = dict(population='full_validation_no_pt_cut', rows=566032,
    macro='six_class_macro_ovr', hbb='X_bb_vs_QCD_pair_probability',
    anchors='old_M0HLT_and_new_OFFLINE_primary_selected_checkpoints',
    checkpoint_reselection=False, final_test_accessed=False, final_test_sealed=True,
    snapshot='committed_new_DE000_epochs_at_analysis_start', original_batching_preserved=True)


def artifact(kind, parents, **fields):
    return with_content_hash(dict(contract=f'LUKA_SIXCLASS_FULL_RECOVERY_{kind}/v1',
        schema_version=1, parents=parents, policy=dict(POLICY), **fields))


def validate(value, kind, parents):
    validate_content_hash(value, expected_contract=f'LUKA_SIXCLASS_FULL_RECOVERY_{kind}/v1')
    check(value['parents'] == parents and value['policy'] == POLICY, 'Analysis binding differs')
    return value


def pooled_metrics(arrays, probabilities):
    from hlt_classification.luka_sixclass_balanced_oscar.reporting import evaluate_probabilities
    y = np.concatenate([arrays[r+'_labels'] for r in hbb.ROLES])
    p = np.concatenate([probabilities[r] for r in hbb.ROLES])
    return dict(macro=evaluate_probabilities(y, p), hbb=hbb.binary_metrics(y, p))


def saved_probabilities(result, arrays):
    values = {}
    for role, key in zip(hbb.ROLES, ('predictions', 'outside_predictions')):
        with np.load(validate_file_ref(result['receipt']['files'][key]), allow_pickle=False) as payload:
            check(set(payload.files) == {'identities', 'labels', 'pt', 'probabilities'},
                  'Prediction inventory differs')
            for field in ('identities', 'labels', 'pt'):
                check(np.array_equal(payload[field], arrays[role+'_'+field]),
                      'Prediction population differs: '+role+'/'+field)
            values[role] = payload['probabilities'].copy()
    return values


def anchor(result, arrays):
    """Use the published primary selection for BOTH metrics, never Hbb reselection."""
    check(result['receipt']['selected_epoch'] == result['report']['selected_pass'],
          'Anchor selected epoch differs')
    return dict(epoch=result['report']['selected_pass'], result=result['receipt']['content_hash'],
                metrics=pooled_metrics(arrays, saved_probabilities(result, arrays)))


def recovery(value, low, high):
    check(all(np.isfinite(v) and 0 <= v <= 1 for v in (value, low, high)), 'Invalid AUC')
    return 100*(value-low)/(high-low) if high > low else None


def snapshot(current, saved=None):
    """Freeze the completed prefix; later training cannot extend a resumed audit."""
    check(1 <= len(current) <= 20 and [r['epoch'] for r in current] == list(range(1, len(current)+1)),
          'Need a contiguous committed DE000 history')
    if saved is not None:
        check(saved and current[:len(saved)] == saved, 'Snapshot changed or lost committed epochs')
        return saved
    return current


def native_from_epoch(family, receipt, device):
    import torch
    from hlt_classification.luka_sixclass_balanced_oscar import model as old_model
    from hlt_classification.luka_sixclass_low_lr_oscar import model as new_model
    check(family in ('Previous', 'New'), 'Unknown DE000 family')
    state = torch.load(validate_file_ref(receipt['state']), map_location='cpu', weights_only=True)
    check(state['epoch'] == receipt['epoch'] and state['update'] == receipt['epoch']*12961,
          'Checkpoint epoch/update differs')
    student = (old_model if family == 'Previous' else new_model).new_model('DE000')
    student.load_state_dict(state['student'], strict=True)
    return student.deployed().to(device).eval()


def evaluate_family(audit, family, commits, caches, arrays, output, *, published=None, device='cuda:0'):
    from hlt_classification.luka_sixclass_balanced_oscar import training, reporting
    from hlt_classification.luka_sixclass_low_lr_oscar.training import binary_hbb
    selected = published['report']['selected_pass'] if published is not None else None
    # Qualify original-selected replay before spending time on the remaining epochs.
    order = sorted(commits, key=lambda r: (r['epoch'] != selected, r['epoch']))
    rows = []
    for commit in order:
        epoch = commit['epoch']
        parents = dict(analysis=audit['content_hash'], epoch_commit=commit['content_hash'])
        target = Path(output)/family/f'{epoch:02d}.json'
        saved = validate(load_json(target), 'EPOCH', parents) if target.exists() else None
        if saved is not None:
            check(saved['family'] == family and saved['epoch'] == epoch, 'Saved row identity differs')
        # First epoch for new family also repeats on resume to qualify this allocation.
        if saved is None or epoch == (selected or 1):
            begun = time.monotonic()
            native = native_from_epoch(family, commit, device)
            probabilities = {r: training.predict(native, caches[r], device) for r in hbb.ROLES}
            check(reporting.evaluate_probabilities(caches['validation'].labels, probabilities['validation'])
                  == commit['row']['validation'], 'In-range metrics did not reproduce')
            if epoch == selected:
                hbb.check_saved_predictions(published, caches, probabilities, arrays)
            if family == 'New':
                check(binary_hbb(np.concatenate([caches[r].labels for r in hbb.ROLES]),
                      np.concatenate([probabilities[r] for r in hbb.ROLES])) == commit['row']['full_hbb'],
                      'Saved full-Hbb metrics did not reproduce')
            metrics = pooled_metrics(arrays, probabilities)
            row = artifact('EPOCH', parents, family=family, epoch=epoch, metrics=metrics,
                in_range_metrics_reproduced=True, selected_predictions_reproduced=epoch == selected,
                saved_full_hbb_reproduced=family == 'New', analysis_job=os.environ['SLURM_JOB_ID'],
                seconds=time.monotonic()-begun)
            if saved is not None:
                check(saved['metrics'] == metrics, 'Resumed metrics differ')
                row = saved
            else:
                target.parent.mkdir(exist_ok=True)
                write_immutable_json(target, row)
            del native, probabilities
        else:
            row = saved
        rows.append(row)
        print(f"{family} DE000 epoch {epoch:02d}: full macro AUC "
              f"{row['metrics']['macro']['macro_ovr_auc']:.8f}; Hbb AUC {row['metrics']['hbb']['auc']:.8f}", flush=True)
    return sorted(rows, key=lambda r:r['epoch'])


def run(spec_path, output, analysis_source):
    import torch
    from hlt_classification.provenance import capture_source_snapshot
    from hlt_classification.luka_new_fullstat_hydra import lineage
    from hlt_classification.luka_sixclass_balanced_oscar import artifacts as old_artifacts, population
    from hlt_classification.luka_sixclass_low_lr_oscar import campaign, artifacts, reuse

    spec = load_json(spec_path)
    campaign.check(spec)
    check(Path(spec_path).resolve() == Path(spec['root'])/'campaign_spec.json', 'Wrong spec locator')
    old = reuse.check(spec['reuse'], payloads=True)
    check(spec['population'] == old['population'] and spec['environment'] == old['environment'],
          'Population/runtime differs between campaigns')
    check(torch.cuda.is_available() and torch.cuda.device_count() == 1
          and torch.cuda.is_bf16_supported() and torch.cuda.get_device_name(0) == 'NVIDIA L40S',
          'Use one L40S with the registered BF16 runtime')
    check(os.environ.get('SLURM_JOB_ID', '').isdigit(), 'Use the GPU analysis allocation')
    check(os.environ.get('PYTHONNOUSERSITE') == '1' and
          os.environ.get('LD_LIBRARY_PATH', '').split(':')[0] == spec['environment_dir']+'/lib',
          'Use the registered runtime helper')
    source = capture_source_snapshot(analysis_source, require_clean=True)
    output = Path(output).resolve()
    check(output.is_relative_to(Path('/oscar/home/rlyang/hlt_classification/reports')),
          'Use a separate reports directory')
    output.mkdir(parents=True, exist_ok=True)
    with hbb.execution_lock(output):
        check(not (output/'report.json').exists(), 'Analysis already published; read report.json')
        torch.set_num_threads(1)
        print('Authenticating saved anchors, original DE000 and committed new epochs...', flush=True)
        old_memo = {}
        old_de = old_artifacts.completed(old, 'DE000', memo=old_memo)
        old_hlt = old_artifacts.completed(old, 'M0HLT', memo=old_memo)
        new_offline = artifacts.completed(spec, 'OFFLINE')
        for result, job in ((old_de, '9198212'), (old_hlt, '9198208'), (new_offline, '9442678')):
            check(result['receipt']['job_id'] == job, 'Unexpected reference job binding')
        dependencies = dict(OFFLINE=new_offline, **{t:old_memo[t] for t in ('CONCAT','UE100','DE050')})
        parents = artifacts.fit_parents(spec, 'DE000', dependencies)
        _, new_commits = artifacts.history(spec, 'DE000', parents, states=False)
        _, old_commits = old_artifacts.history(old, 'DE000', old_de['receipt']['fit_parents'], states=False)
        check(len(old_commits) == 20, 'Old DE000 must have all 20 epochs')
        audit_path = output/'analysis_spec.json'
        existing = load_json(audit_path) if audit_path.exists() else None
        new_commits = snapshot(new_commits, existing['epochs']['New'] if existing else None)
        for commit in new_commits:
            validate_file_ref(commit['state'])
        check(lineage.verify(spec['dataset'], spec['project'], spec['population']['parents']['foundation'])
              == spec['foundation_producer'], 'Population producer changed')
        loaded = population.load(spec['dataset'], spec['population_root'])
        check(loaded[0] == spec['population'], 'Registered population differs')
        proof = hbb.check_population(loaded[4])
        anchors = dict(M0HLT=anchor(old_hlt, loaded[4]), OFFLINE=anchor(new_offline, loaded[4]))
        audit = artifact('SPEC', dict(new_campaign=spec['content_hash'], old_campaign=old['content_hash'],
            source=source['content_hash']), source=source, new_source=spec['source'], old_source=old['source'],
            runtime=spec['environment'], population=proof, anchors=anchors,
            epochs=dict(Previous=old_commits, New=new_commits))
        if existing is not None:
            check(existing == audit, 'Resume analysis specification differs')
        else:
            write_immutable_json(audit_path, audit)
        print(f"Snapshot: {len(new_commits)}/20 new DE000 epochs; old DE000 has 20.", flush=True)
        caches = hbb.load_validation(old, loaded, 'DE000')
        histories = {}
        for family, commits, result in (('Previous', old_commits, old_de), ('New', new_commits, None)):
            histories[family] = evaluate_family(audit, family, commits, caches, loaded[4], output, published=result)
        for folder, want in ((spec['project'],spec['source']), (old['project'],old['source']), (analysis_source,source)):
            check(capture_source_snapshot(folder, require_clean=True) == want, 'Source changed during analysis')
        report = artifact('REPORT', dict(analysis=audit['content_hash']), anchors=anchors,
                          population=proof, epoch_history=histories, new_epochs=len(new_commits))
        write_immutable_json(output/'report.json', report)
        print_report(report)
        print('Saved:', output/'report.json', flush=True)


def print_report(report):
    print('\nFull validation: 566,032 jets (328,589 in range + 237,443 outside).')
    print('Natural validation; six classes; no pT cut or checkpoint reselection.')
    print('Recovery anchors: old M0HLT and new OFFLINE, fixed primary-selected checkpoints.')
    for name, key, metric in (('SIX-CLASS MACRO OVR AUC','macro','macro_ovr_auc'),
                               ('HBB-VERSUS-QCD AUC','hbb','auc')):
        print('\n'+name)
        lo, hi = [report['anchors'][s]['metrics'][key][metric] for s in ('M0HLT','OFFLINE')]
        for stage, value in (('M0HLT',lo),('OFFLINE',hi)):
            print(f"  {stage} anchor: epoch {report['anchors'][stage]['epoch']}, AUC {value:.8f}")
        if key == 'hbb':
            print('  Binary sample: 26,937 X_bb + 432,708 QCD; score p(bb)/(p(bb)+p(QCD)).')
        print(f"{'Epoch':>5} {'Prev DE000':>12} {'Rec.%':>9} {'New DE000':>12} {'Rec.%':>9} {'Delta AUC':>12}")
        lookup = {s:{r['epoch']:r['metrics'][key][metric] for r in report['epoch_history'][s]}
                  for s in ('Previous','New')}
        for epoch in range(1,21):
            old_value, new_value = (lookup[s].get(epoch) for s in ('Previous','New'))
            fields = []
            for value in (old_value,new_value):
                rec = recovery(value,lo,hi) if value is not None else None
                fields.extend([f'{value:12.8f}' if value is not None else f"{'-':>12}",
                               f'{rec:9.2f}' if rec is not None else f"{'n/a':>9}"])
            delta = f'{new_value-old_value:+12.8f}' if new_value is not None and old_value is not None else f"{'-':>12}"
            print(f'{epoch:5d} '+ ' '.join(fields)+' '+delta)
    print(f"\nNew DE000 snapshot: {report['new_epochs']}/20 completed epochs at analysis start.")
    print('Recovery = 100*(DE000 - old M0HLT)/(new OFFLINE - old M0HLT), separately for each metric.')
    print('Nonpositive anchor gap: n/a. Both DE000 columns use these same anchors.')
    print('Validation-only replay; no training, test inference, or original file/job changes.')
