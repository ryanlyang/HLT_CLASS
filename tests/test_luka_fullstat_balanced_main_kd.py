"""Direct deployed-head KD, unchanged initialization and native inference."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from test_luka_fullstat_balanced_ladder import threads
from hlt_classification.luka_fullstat_balanced_ladder import contracts as old, model as original
from hlt_classification.luka_fullstat_balanced_main_kd import contracts as c, model, source
from hlt_classification.luka_u100_multiview.teachers import forward


def test_only_authorized_recipe_changes_and_initialization(monkeypatch):
    from test_luka_embedding_aux import tiny
    tiny(monkeypatch)
    a,b=old.recipe(),c.recipe()
    ignore={'content_hash','contract','base_recipe_sha256','main_kd_weight','main_kd_teacher',
            'main_head_supervision','ce_weight','teacher_order','coordinates'}
    assert {k:v for k,v in a.items() if k not in ignore}=={k:v for k,v in b.items() if k not in ignore}
    assert (b['ce_weight'],b['main_kd_weight'],b['main_kd_teacher'])==(.25,.75,'DE050')
    assert b['base_recipe_sha256']==a['content_hash'] and c.tasks()=={'train_DE000':[]}
    assert c.node('DE000')==dict(old.node('DE000'),main_logit_teacher='DE050')
    before=original.new_model('DE000');rng=torch.get_rng_state().clone()
    after=model.new_model('DE000')
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,after.state_dict()[k]) for k,v in before.state_dict().items())
    with pytest.raises(PermissionError):model.new_model('DE050')
    with pytest.raises(ValueError):c.validate(a,'RECIPE')


def loss_fixture(device):
    from test_luka_attention import cache
    data=cache(coordinate='D000',rows=22);ids=np.array([1,1,2,3,4]);rng=np.random.default_rng(43)
    banks={}
    for t in c.TEACHERS['DE000']:
        probs=rng.dirichlet(np.ones(11)*.2,size=22).astype(np.float32)
        banks[t]=SimpleNamespace(targets=rng.normal(size=(22,128)).astype(np.float32),probabilities=probs)
    keys=F.normalize(torch.randn(22,128,device=device),dim=-1)
    match=SimpleNamespace(candidate_keys=lambda ix:keys[ix])
    candidates=np.tile(np.arange(1025)%22,(len(ids),1));candidates[:,0]=ids
    return data,ids,banks,match,candidates


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_main_gradient_exact_mixture_and_removable_auxiliaries(monkeypatch,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    if device=='cpu':
        from test_luka_embedding_aux import tiny
        tiny(monkeypatch)
    data,ids,banks,match,candidates=loss_fixture(device)
    if device=='cuda':
        from hlt_classification.jetclass2_delphes.acceptance import installed_parity
        from hlt_classification.jetclass2_delphes.dzfix_fusion_model import parity_backend
        with parity_backend(device):installed_parity(data,device=device)
    net=model.new_model('DE000').to(device).eval()
    loss,parts=model.objective(net,data.batch(ids),ids,banks,update=6,steps=2,device=device,
                               sampler=match,candidates=candidates)
    logits=forward(net,data.batch(ids),device,bf16=device=='cuda').float()
    teacher=torch.tensor(banks['DE050'].probabilities[ids],device=device)
    oracle=(teacher*(teacher.log()-F.log_softmax(logits/2.,dim=-1))).sum(-1).mean()*4
    torch.testing.assert_close(parts['main_kd'],oracle)
    expected=.25*parts['ce']+.75*oracle+.3*parts['matching']
    for t in c.TEACHERS['DE000']:expected=expected+2.5*parts['embedding_'+t]+.25*parts['kd_'+t]
    torch.testing.assert_close(loss,expected)
    main=list(net.mod.fc.parameters())
    ce=torch.autograd.grad(parts['ce'],main,retain_graph=True)
    kd=torch.autograd.grad(parts['main_kd'],main,retain_graph=True)
    exact=torch.autograd.grad(.25*parts['ce']+.75*parts['main_kd'],main,retain_graph=True)
    assert sum(g.square().sum() for g in kd)>0
    assert any(not torch.allclose(g,h) for g,h in zip(exact,ce))
    loss.backward()
    for p,g in zip(main,exact):torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    native=net.deployed().to(device).eval();net.capture=False
    with torch.inference_mode():
        expected=forward(native,data.batch(ids),device,bf16=False)
        for head in (*net.predictors.values(),*net.logit_heads.values(),net.matching):
            for p in head.parameters():p.fill_(float('nan'))
        torch.testing.assert_close(expected,forward(net,data.batch(ids),device,bf16=False),rtol=0,atol=0)
    assert not any(k.startswith(('predictors.','logit_heads.','matching.')) for k in native.state_dict())


def test_original_runtime_import_closure_is_unchanged_from_donor():
    from pathlib import Path
    import subprocess,hashlib
    project=Path(__file__).resolve().parents[1];files=source.closure(project)
    assert 'src/hlt_classification/luka_fullstat_balanced_ladder/model.py' in files
    assert 'src/hlt_classification/luka_fullstat_balanced_ladder/occupancy.json' in files
    for name in files:
        original=subprocess.run(['git','show',source.PARENT_COMMIT+':'+name],cwd=project,check=True,capture_output=True).stdout
        assert hashlib.sha256(original).digest()==hashlib.sha256((project/name).read_bytes()).digest(),name
