"""Chunked reconstruction with the unchanged complete-support hard loss."""
import time
import torch
from hlt_classification.luka_next_supervision_hydra import reconstruction_batched as hard


def backward(decoder,h,tokens,mask,targets,weight,*,chunk=8):
    sh=h.detach().requires_grad_(True);st=tokens.detach().requires_grad_(True)
    total=0.;statistics={};started=time.perf_counter()
    for start in range(0,len(h),chunk):
        end=min(start+chunk,len(h));scores=decoder(sh[start:end],st[start:end],mask[start:end])
        losses,parts=hard.rows(scores,targets[start:end]);parts={k:float(v.detach().mean()) for k,v in parts.items()}
        loss=losses.sum()/len(h);(weight*loss).backward();total+=float(loss.detach())
        for key,value in parts.items():
            statistics[key]=max(statistics.get(key,0.),value) if key.endswith('_max') else statistics.get(key,0.)+value*(end-start)/len(h)
        del scores,losses,loss
    if sh.grad is None or st.grad is None:raise ValueError('Disconnected reconstruction condition')
    return total,statistics,sh.grad,st.grad,dict(decoder_seconds=time.perf_counter()-started)
