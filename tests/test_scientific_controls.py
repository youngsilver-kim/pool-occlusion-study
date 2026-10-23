"""Tests protect experimental validity, not implementation-shaped snapshots."""
import copy
import numpy as np
import pytest
import torch
from poolstudy.common import DEFAULT, METHODS
from poolstudy.data import make_split, Images, corrupt, conditions
from poolstudy.models import Classifier, PoolHead, normalized_entropy
from poolstudy.metrics import classification_metrics, mean_interval, holm_adjust

@pytest.fixture
def cfg():
    c=copy.deepcopy(DEFAULT)
    c.update(dataset="cifar10",synthetic=True)
    return c

def test_split_is_disjoint_stratified_and_repeatable(cfg):
    y=np.repeat(np.arange(10),20)
    train,val=make_split(y,cfg)
    assert not set(train)&set(val)
    assert set(train)|set(val)==set(range(len(y)))
    assert np.array_equal(np.bincount(y[val]),np.full(10,2))
    assert all(np.array_equal(a,b) for a,b in zip((train,val),make_split(y,cfg)))

def test_backbone_and_classifier_are_paired_across_all_methods(cfg):
    ref=Classifier(cfg,"attention",13)
    for method in METHODS:
        model=Classifier(cfg,method,13)
        for a,b in zip(ref.backbone.state_dict().values(),model.backbone.state_dict().values()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(ref.fc.weight,model.fc.weight,rtol=0,atol=0)
        if method.startswith("attention") or method=="mixed_attention":
            torch.testing.assert_close(ref.pool.score.weight,model.pool.score.weight,rtol=0,atol=0)

def test_augmentation_does_not_depend_on_loader_order(cfg):
    x=np.random.default_rng(1).integers(0,256,(8,32,32,3),dtype=np.uint8)
    y=np.arange(8)
    a=Images(x,y,np.arange(8),cfg,train=True,seed=13,epoch=5)
    b=Images(x,y,np.arange(8)[::-1],cfg,train=True,seed=13,epoch=5)
    torch.testing.assert_close(a[3][0],b[4][0],rtol=0,atol=0)

def test_corruptions_shared_across_methods_and_training_seeds(cfg):
    image=np.ones((32,32,3),dtype=np.float32)*.7
    cond=next(c for c in conditions(cfg) if c["kind"]=="occlusion")
    a,mask=corrupt(image,19,cond,cfg)
    other=copy.deepcopy(cfg); other["seeds"]=[999]; other["methods"]=["gap"]
    b,mask2=corrupt(image,19,cond,other)
    assert np.array_equal(a,b) and np.array_equal(mask,mask2)
    side=round(32*np.sqrt(cond["area"]))
    assert mask.sum()==side*side
    assert np.all(a[mask==0]==image[mask==0])

def test_floor_is_inactive_above_threshold_and_backpropagates_below(cfg):
    head=PoolHead(4,"attention_floor",cfg).train()
    x=torch.zeros(2,4,8,8)
    _,penalty,w=head(x)
    assert penalty.item()==0
    torch.testing.assert_close(normalized_entropy(w),torch.ones(2))
    with torch.no_grad():
        head.score.weight.fill_(.5)
    x[:,:,0,0]=3
    _,penalty,_=head(x)
    assert penalty.item()>0
    penalty.backward()
    assert head.score.weight.grad is not None
    assert torch.isfinite(head.score.weight.grad).all()
    assert head.score.weight.grad.abs().sum()>0

def test_all_methods_have_finite_backward_and_normalized_maps(cfg):
    torch.set_num_threads(2)
    x=torch.rand(2,3,32,32)
    for method in METHODS:
        model=Classifier(cfg,method,13).train()
        logits,penalty,weights=model(x,diagnostics=True)
        (torch.nn.functional.cross_entropy(logits,torch.tensor([0,1]))+penalty).backward()
        assert logits.shape==(2,10)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if weights is not None:
            torch.testing.assert_close(weights.sum(-1),torch.ones(2))

def test_calibration_and_uncertainty_use_known_answers():
    m,_=classification_metrics(np.array([[.8,.2],[.1,.9]]),np.array([0,1]),bins=2)
    assert m["accuracy"]==100
    assert m["ece"]==pytest.approx(.15)
    assert m["brier"]==pytest.approx(.05)
    assert mean_interval([1])["sd"] is None
    assert mean_interval([1,1,1])["ci_low"]==1
    assert np.allclose(holm_adjust([.01,.04,.03]),[.03,.06,.06])

def test_epoch_resume_recovers_committed_checkpoint_and_history(cfg,tmp_path,monkeypatch):
    import pandas as pd
    from poolstudy import engine
    from poolstudy.data import load_raw
    torch.set_num_threads(2)
    cfg.update(epochs=2,warmup_epochs=1,lr=.01,batch_size=16,eval_batch_size=32,
               synthetic_train_per_class=4,output_dir=str(tmp_path),name="continuous")
    raw=load_raw(cfg,train=True)
    device=torch.device("cpu")
    method,seed="attention_entropy",13
    engine.train_one(cfg,method,seed,device,raw)
    reference=engine.load_checkpoint(engine.run_dir(cfg,method,seed)/"last.pt")
    cfg=copy.deepcopy(cfg); cfg["name"]="interrupted"
    original_save=engine.save_checkpoint
    def fail_after_commit(path,payload):
        original_save(path,payload)
        raise RuntimeError("simulated crash immediately after checkpoint commit")
    monkeypatch.setattr(engine,"save_checkpoint",fail_after_commit)
    with pytest.raises(RuntimeError,match="simulated crash"):
        engine.train_one(cfg,method,seed,device,raw)
    monkeypatch.setattr(engine,"save_checkpoint",original_save)
    engine.train_one(cfg,method,seed,device,raw)
    resumed=engine.load_checkpoint(engine.run_dir(cfg,method,seed)/"last.pt")
    for name,weight in reference["model"].items():
        torch.testing.assert_close(weight,resumed["model"][name],rtol=0,atol=0)
    path=engine.run_dir(cfg,method,seed)/"history.csv"
    path.unlink()
    # A completed checkpoint must also regenerate a missing history file.
    engine.train_one(cfg,method,seed,device,raw)
    assert pd.read_csv(path).epoch.tolist()==[1,2]
