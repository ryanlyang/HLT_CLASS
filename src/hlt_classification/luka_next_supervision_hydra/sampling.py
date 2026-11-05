"""Identity-seeded, parent-first TRAIN sampling; no IDs enter a neural head."""
import hashlib
import json
import math
from collections import OrderedDict, Counter
import numpy as np
import torch
from .contracts import recipe,GROUP_PROPOSALS
from .data import pair_features
from hlt_classification.luka_u100_contrastive.sampling import Sampler as JetSampler, fit_bins


def quotas(n):
    return [n//4+(j<n%4) for j in range(4)]


def signatures(local, member_mask=None):
    x=local
    discrete=np.column_stack((x[:,6:12].argmax(-1),x[:,5]+1,
        (x[:,18:22]*np.array([1,2,4,8])).sum(-1))).astype(np.int64)
    numeric=np.column_stack((np.searchsorted(recipe()['pt_edges'],x[:,0]),
        np.searchsorted(recipe()['radius_edges'],np.hypot(x[:,2],np.arctan2(x[:,3],x[:,4]))))).astype(np.int64)
    return discrete,numeric


class ParentPositions:
    """Sample the sorted eligible-parent positions without a population-size mask.

    This is the same ordered population and RNG draw as choice(flatnonzero(mask)).
    Only the few excluded positions are materialized, including exhausted parents.
    """
    def __init__(self, parents, forbidden):
        values=np.array(sorted(forbidden),np.int64)
        positions=np.searchsorted(parents,values)
        live=positions<len(parents)
        positions=positions[live]; values=values[live]
        self.blocked=positions[parents[positions]==values]
        self.shifted=self.blocked-np.arange(len(self.blocked))
        self.size=len(parents)-len(self.blocked)

    def __len__(self): return self.size

    def draw(self, rng, number):
        virtual=rng.integers(self.size,size=number)
        return virtual+np.searchsorted(self.shifted,virtual,side='right')


class MembershipSampler:
    def __init__(self, particles, pt, count, *, replicate=0, event_groups=None):
        self.p=particles; self.replicate=replicate; self.stream='training'
        self.bins,self.edges=fit_bins(pt,count); self.labels=particles.labels
        if len(self.bins)!=len(self.labels): raise ValueError('Sampling summaries do not join TRAIN')
        self.classes=[np.flatnonzero(self.labels==c) for c in range(11)]
        if any(len(x)<32 for x in self.classes): raise ValueError('Need at least32 TRAIN parents per class')
        self.events=None if event_groups is None else np.asarray(event_groups)
        if self.events is not None and self.events.shape!=self.labels.shape: raise ValueError('Malformed event groups')
        # These are bounded categorical/bin indices, not model measurements.
        # Avoid ever materializing full TRAIN int64 descriptor intermediates.
        self.d=np.empty((len(particles.local),3),np.int8); self.n=np.empty((len(particles.local),2),np.int8)
        for start in range(0,len(self.d),65536):
            stop=min(start+65536,len(self.d))
            self.d[start:stop],self.n[start:stop]=signatures(particles.local[start:stop])
        self.sizes=np.diff(particles.offsets)
        self.maximum_radius=int(np.ptp(self.bins,axis=0).max())
        self.feature_radius=int(np.ptp(self.n,axis=0).max())
        self.capacity=dict(class_counts=[len(x) for x in self.classes],event_groups_available=self.events is not None,
            parent_cap=16,group_parent_cap=8,source=particles.source,rows=len(self.labels))
        # Fixed mixed-radix codes have the same lexicographic order as v1's
        # five-column int64 unique, without its full-size sorting temporaries.
        shape=(6,3,16,len(recipe()['pt_edges'])+1,len(recipe()['radius_edges'])+1)
        codes=np.empty(len(self.d),np.uint16); counts=np.zeros(math.prod(shape),np.int64)
        for start in range(0,len(codes),65536):
            stop=min(start+65536,len(codes)); keys=np.column_stack((self.d[start:stop],self.n[start:stop]))
            chunk=np.ravel_multi_index(keys.T,shape)
            codes[start:stop]=chunk; counts+=np.bincount(chunk,minlength=len(counts))
        order=np.argsort(codes,kind='stable'); cuts=np.r_[0,counts.cumsum()]; present=np.flatnonzero(counts)
        self.cell_keys=np.array(np.unravel_index(present,shape)).T
        self.cells={}; self.exact_regions={}
        for cell,code in enumerate(present):
            rows=order[cuts[code]:cuts[code+1]]; self.cells[cell]=rows
            parents=self.p.parents[rows]; starts=np.r_[0,np.flatnonzero(np.diff(parents))+1]
            self.exact_regions[tuple(self.cell_keys[cell])]=(rows,parents[starts],starts,np.diff(np.r_[starts,len(rows)]))
        self.regions=OrderedDict(); self.region_bytes=0
        self.parent_rows=np.arange(len(self.labels)); self.rank=np.argsort(np.argsort(
            np.array([bytes(x).hex() for x in particles.identities]),kind='stable'),kind='stable')
        self.parent_pools={}; self.exclusion_cache={}; self.kinematic_cells={}
        self.kinematic_regions={}
        for g in (1,2,3):
            eligible=self.parent_rows[np.diff(self.p.offsets)>=g]
            for cell in np.unique(self.bins,axis=0):
                self.kinematic_cells[g,tuple(cell)]=eligible[(self.bins[eligible]==cell).all(-1)]

    def rng(self,epoch,anchor,component,group_size=1):
        if type(epoch) is not int or epoch<1: raise ValueError('Epoch is one-based')
        key=['GROUP_A_MEMBERSHIP/v1',2026110508,self.replicate,epoch,
             bytes(self.p.identities[anchor]).hex(),self.stream,self.p.source,group_size,component]
        return np.random.default_rng(int.from_bytes(hashlib.sha256(json.dumps(key,separators=(',',':')).encode()).digest(),'little'))

    def capacity_proof(self,family):
        """Sufficient population-wide bounds, before any scientific iteration."""
        sizes=np.diff(self.p.offsets); bounds={}
        supports=(1,) if family=='P' else (1,2,3)
        for g in supports:
            cap=16 if family=='P' else 8
            available=np.minimum(cap,sizes) if family=='P' else np.array([min(cap,math.comb(int(n),g)) if n>=g else 0 for n in sizes])
            maximum=available.max()
            excluded=np.full(11,maximum,np.int64)
            global_excluded=int(maximum)
            if self.events is not None:
                _,inverse=np.unique(self.events,return_inverse=True)
                sums=np.zeros((inverse.max()+1,11),np.int64)
                np.add.at(sums,(inverse,self.labels),available)
                excluded=sums.max(0); global_excluded=int(sums.sum(-1).max())
            perclass=np.bincount(self.labels,weights=available,minlength=11)-excluded
            # The4096 M proposal visits G1024 before S1024; account for the
            # maximum previous same-class draw and the separate final pool.
            needed=2547 if family=='P' else 127
            active=np.array([np.any(sizes[rows]>=g) for rows in self.classes])
            if np.any(perclass[active]<needed): raise ValueError('Insufficient same-class unique/capped TRAIN capacity')
            if family=='P' and available.sum()-global_excluded-499<4096:
                raise ValueError('Insufficient full semi-hard proposal capacity')
            bounds[str(g)]=dict(minimum_class_capacity=int(perclass[active].min()) if active.any() else None,
                required=needed,global_capacity=int(available.sum()-global_excluded),maximum_excluded=global_excluded)
        return dict(self.capacity,family=family,bounds=bounds)

    def excluded(self,anchor):
        if self.events is None: return {int(anchor)}
        if anchor not in self.exclusion_cache:
            self.exclusion_cache[anchor]=set(np.flatnonzero(self.events==self.events[anchor]).tolist())|{int(anchor)}
        return self.exclusion_cache[anchor]

    def region(self,prototype,discrete_distance,radius):
        if discrete_distance==0 and radius==0:
            # Built once before fork, shared read-only by every worker. Exact
            # cells partition the particles rather than duplicating regions.
            return self.exact_regions[(*self.d[prototype],*self.n[prototype])]
        key=(*self.d[prototype],*self.n[prototype],discrete_distance,radius)
        if key not in self.regions:
            delta=(self.cell_keys[:,:3]!=self.d[prototype]).sum(-1)
            numeric=np.abs(self.cell_keys[:,3:]-self.n[prototype]).max(-1)
            cells=np.flatnonzero((delta==discrete_distance)&(numeric<=radius))
            rows=np.sort(np.concatenate([self.cells[int(c)] for c in cells])) if len(cells) else np.empty(0,np.int64)
            # Sorted by global particle index, hence contiguous by parent.
            parents,starts,counts=np.unique(self.p.parents[rows],return_index=True,return_counts=True)
            value=(rows,parents,starts,counts); self.regions[key]=value
            self.region_bytes+=sum(a.nbytes for a in value)
            while self.region_bytes>64*2**20 and len(self.regions)>1:
                _,old=self.regions.popitem(last=False); self.region_bytes-=sum(a.nbytes for a in old)
        self.regions.move_to_end(key)
        return self.regions[key]

    def parent(self,anchor,law,rng,used,cap,g):
        forbidden=self.excluded(anchor)|{p for p,n in used.items() if n>=cap}
        # The base eligibility is independent of the K cell; radius-specific
        # arrays live in the separately bounded kinematic-region cache.
        key=(g,law,int(self.labels[anchor]) if law=='S' else None)
        if key not in self.parent_pools:
            base=self.parent_rows[np.diff(self.p.offsets)>=g]
            if law=='S': base=base[self.labels[base]==self.labels[anchor]]
            self.parent_pools[key]=base
        base=self.parent_pools[key]
        if law=='K':
            for radius in range(self.maximum_radius+1):
                pool=self.kinematic_region(anchor,g,radius)
                for _ in range(min(32,len(pool))):
                    p=int(pool[rng.integers(len(pool))])
                    if p not in forbidden: self.last_kinematic_radius=radius; return p
                pool=pool[~np.isin(pool,list(forbidden))]
                if len(pool): self.last_kinematic_radius=radius; return int(rng.choice(pool))
            raise ValueError('Kinematic capacity exhausted')
        if law=='G':
            # Uniform source class, then uniform parent; rejection preserves the
            # conditional uniform law without scanning the population per draw.
            classes=list(range(11))
            while classes:
                c=int(rng.choice(classes)); pool=self.classes[c]
                for _ in range(64):
                    p=int(rng.choice(pool))
                    if p not in forbidden and self.p.offsets[p+1]-self.p.offsets[p]>=g: return p
                pool=pool[(np.diff(self.p.offsets)[pool]>=g)&~np.isin(pool,list(forbidden))]
                if len(pool): return int(rng.choice(pool))
                classes.remove(c)
        else:
            for _ in range(64):
                p=int(rng.choice(base)) if len(base) else -1
                if p>=0 and p not in forbidden: return p
            pool=base[~np.isin(base,list(forbidden))]
            if len(pool): return int(rng.choice(pool))
        raise ValueError('Declared parent capacity exhausted')

    def kinematic_region(self,anchor,g,radius):
        key=(g,*self.bins[anchor],radius)
        if key not in self.kinematic_regions:
            rows=[rows for (size,cell),rows in self.kinematic_cells.items()
                if size==g and max(abs(np.asarray(cell)-self.bins[anchor]))<=radius and len(rows)]
            self.kinematic_regions[key]=np.concatenate(rows) if rows else np.empty(0,np.int64)
            if sum(a.nbytes for a in self.kinematic_regions.values())>64*2**20:
                # Bound this optimization independently of the scientific law.
                saved=self.kinematic_regions[key]; self.kinematic_regions={key:saved}
        return self.kinematic_regions[key]

    def feature_particle(self,prototype,rng,blocked,used,cap,excluded):
        maximum=self.feature_radius
        for discrete in range(4):
            for radius in range(maximum+1):
                rows,parents,starts,counts=self.region(prototype,discrete,radius)
                eligible=[]
                for _ in range(min(64,len(parents))):
                    j=int(rng.integers(len(parents))); p=int(parents[j])
                    if p in excluded or used.get(p,0)>=cap: continue
                    group=rows[starts[j]:starts[j]+counts[j]]
                    candidates=group[~np.isin(group,list(blocked))]
                    if len(candidates): return int(rng.choice(candidates)),dict(discrete_expansion=discrete,numeric_radius=radius)
                for j,p in enumerate(parents):
                    if int(p) in excluded or used.get(int(p),0)>=cap: continue
                    group=rows[starts[j]:starts[j]+counts[j]]
                    if any(int(x) not in blocked for x in group): eligible.append(j)
                if eligible:
                    j=int(rng.choice(eligible)); candidates=rows[starts[j]:starts[j]+counts[j]]
                    candidates=candidates[~np.isin(candidates,list(blocked))]
                    return int(rng.choice(candidates)),dict(discrete_expansion=discrete,numeric_radius=radius)
        raise ValueError('Feature-matched capacity exhausted; no fallback law')

    def negatives(self,anchor,n,epoch,positives,*,law='M',blocked=None,used=None):
        if law not in ('M','G','S','K','F'): raise ValueError('Unregistered particle negative law')
        blocked=set() if blocked is None else set(blocked); used=Counter() if used is None else Counter(used)
        selected=[]; details=[]; excluded=self.excluded(anchor)
        exhausted={p for p,v in used.items() if v>=16}
        laws=list('GSKF') if law=='M' else [law]
        ns=quotas(n) if law=='M' else [n]
        for component,number in zip(laws,ns):
            rng=self.rng(epoch,anchor,'negative_'+component)
            def accept(idx,info):
                parent=int(self.p.parents[idx]); selected.append(idx); blocked.add(idx); used[parent]+=1
                if used[parent]>=16: exhausted.add(parent)
                details.append(dict(component=component,**info))
            def bulk(indices,info):
                indices=np.asarray(indices,np.int64); pp=self.p.parents[indices]
                parents,counts=np.unique(pp,return_counts=True)
                if (len(np.unique(indices))!=len(indices) or excluded.intersection(parents.tolist())
                        or blocked.intersection(indices.tolist())
                        or any(used[int(p)]+int(n)>min(16,int(self.sizes[p])) for p,n in zip(parents,counts))): return False
                values=indices.tolist(); selected.extend(values); blocked.update(values)
                used.update({int(p):int(n) for p,n in zip(parents,counts)})
                exhausted.update(int(p) for p in parents if used[int(p)]>=16)
                details.extend([dict(component=component,**info)]*len(indices))
                return True
            if component=='F':
                prototypes=rng.choice(positives,number)
                # Group identical strata. Parent draws are vectorized, retaining
                # uniform parents rather than weighting them by token count.
                grouped={}
                for proto in prototypes:
                    grouped.setdefault((*self.d[proto],*self.n[proto]),[int(proto),0])[1]+=1
                maximum=self.feature_radius
                for proto,need in grouped.values():
                    for discrete in range(4):
                        for radius in range(maximum+1):
                            rows,parents,starts,counts=self.region(proto,discrete,radius)
                            unavailable=set(excluded)
                            positions=ParentPositions(parents,unavailable|exhausted)
                            while need and len(positions):
                                js=positions.draw(rng,max(16,need*2))
                                slots=starts[js]+(rng.random(len(js))*counts[js]).astype(np.int64)
                                if bulk(rows[slots[:need]],dict(discrete_expansion=discrete,numeric_radius=radius)):
                                    need=0; break
                                for j,slot in zip(js,slots):
                                    p=int(parents[j])
                                    if p in unavailable or used[p]>=16: continue
                                    idx=int(rows[slot])
                                    if idx in blocked:
                                        available=[int(x) for x in rows[starts[j]:starts[j]+counts[j]] if int(x) not in blocked]
                                        if not available: unavailable.add(p); continue
                                        idx=available[int(rng.integers(len(available)))]
                                    accept(idx,dict(discrete_expansion=discrete,numeric_radius=radius)); need-=1
                                    if not need: break
                                positions=ParentPositions(parents,unavailable|exhausted)
                            if not need: break
                        if not need: break
                    if need: raise ValueError('Feature-matched capacity exhausted; no fallback law')
            else:
                classes=rng.integers(0,11,number) if component=='G' else np.full(number,int(self.labels[anchor]))
                # Classes are sampled before parents, so collision rejection
                # cannot silently reweight the broad source-class distribution.
                for cls in np.unique(classes):
                    need=int(np.count_nonzero(classes==cls)); radius=0
                    while need:
                        pool=self.kinematic_region(anchor,1,radius) if component=='K' else self.classes[int(cls)]
                        if not len(pool): radius+=1; continue
                        parents=pool[rng.integers(0,len(pool),max(16,need*2))]
                        slots=self.p.offsets[parents]+(rng.random(len(parents))*self.sizes[parents]).astype(np.int64)
                        if bulk(slots[:need],dict(kinematic_radius=radius) if component=='K' else {}): need=0; break
                        before=need
                        for p,idx in zip(parents,slots):
                            p=int(p); idx=int(idx)
                            if p in excluded or used[p]>=min(16,int(self.p.offsets[p+1]-self.p.offsets[p])): continue
                            if idx in blocked:
                                available=[x for x in range(int(self.p.offsets[p]),int(self.p.offsets[p+1])) if x not in blocked]
                                if not available: continue
                                idx=available[int(rng.integers(len(available)))]
                            accept(idx,dict(kinematic_radius=radius) if component=='K' else {}); need-=1
                            if not need: break
                        if need==before:
                            available=[int(p) for p in pool if int(p) not in excluded and used[int(p)]<min(16,int(self.p.offsets[p+1]-self.p.offsets[p]))]
                            if not available:
                                if component=='K' and radius<self.maximum_radius: radius+=1
                                else: raise ValueError('Negative parent capacity exhausted')
        return np.asarray(selected,np.int64),details

    def particle_pool(self,anchor,epoch,law,mine=None):
        if law=='Hsemi':
            if mine is None: raise ValueError('Detached scorer required for semi-hard mining')
            prepared=self.semi_prepare(anchor,epoch)
            return self.semi_finish(prepared,mine(self.p.features[prepared['proposal']]))
        rng=self.rng(epoch,anchor,'positives'); own=np.arange(self.p.offsets[anchor],self.p.offsets[anchor+1])
        positive=rng.choice(own,min(32,len(own)),replace=False); n=500-len(positive)
        negative,info=self.negatives(anchor,n,epoch,positive,law=law)
        return self.particle_record(anchor,epoch,positive,negative,info,{})

    def semi_prepare(self,anchor,epoch):
        rng=self.rng(epoch,anchor,'positives'); own=np.arange(self.p.offsets[anchor],self.p.offsets[anchor+1])
        positive=rng.choice(own,min(32,len(own)),replace=False); n=500-len(positive)
        broad,info=self.negatives(anchor,n//2,epoch,positive)
        proposal,proposal_info=self.negatives(anchor,4096,epoch,positive,blocked=set(broad))
        return dict(anchor=anchor,epoch=epoch,positive=positive,broad=broad,proposal=proposal,info=info,
            proposal_components=dict(Counter(x['component'] for x in proposal_info)))

    def semi_finish(self,prepared,scores):
        anchor,epoch,positive,broad,proposal=(prepared[k] for k in ('anchor','epoch','positive','broad','proposal'))
        scores=np.asarray(scores,np.float32)
        if scores.shape!=(4096,) or not np.isfinite(scores).all(): raise ValueError('Invalid detached proposal scores')
        used=Counter(self.p.parents[broad]); n=500-len(positive)
        tie=self.rank[self.p.parents[proposal]]*(int(self.sizes.max())+1)+(proposal-self.p.offsets[self.p.parents[proposal]])
        ranking=np.lexsort((tie,-scores)); band=ranking[256:1024]; chosen=[]
        for pos in self.rng(epoch,anchor,'semi_select').permutation(band):
            idx=int(proposal[pos]); parent=int(self.p.parents[idx])
            if used[parent]>=16: continue
            chosen.append(idx); used[parent]+=1
            if len(chosen)==n-len(broad): break
        if len(chosen)!=n-len(broad): raise ValueError('Fixed mining band cannot satisfy caps; no rank widening')
        return self.particle_record(anchor,epoch,positive,np.r_[broad,chosen],prepared['info'],
            dict(proposals=4096,band=[257,1024],proposal_components=prepared['proposal_components']))

    def particle_record(self,anchor,epoch,positive,negative,info,evidence):
        indices=np.r_[positive,negative]; labels=np.arange(500)<len(positive)
        perm=self.rng(epoch,anchor,'pool_shuffle').permutation(500)
        if len(np.unique(indices))!=500 or set(self.p.parents[negative])&self.excluded(anchor): raise ValueError('Invalid particle pool')
        if max(Counter(self.p.parents[negative]).values())>16: raise ValueError('Parent cap exceeded')
        return dict(indices=indices[perm],labels=labels[perm],evidence=dict(positives=len(positive),
            negative_parents=len(np.unique(self.p.parents[negative])),components=dict(Counter(x['component'] for x in info)),
            max_discrete_expansion=max((x.get('discrete_expansion',0) for x in info),default=0),
            max_numeric_radius=max((x.get('numeric_radius',0) for x in info),default=0),
            max_kinematic_radius=max((x.get('kinematic_radius',0) for x in info),default=0),**evidence))

    def group_pool(self,anchor,epoch,gmax):
        count=int(self.p.offsets[anchor+1]-self.p.offsets[anchor]); g=min(gmax,count)
        rng=self.rng(epoch,anchor,'groups',gmax); positive=[]; seen=set()
        for _ in range(10000):
            value=tuple(sorted(rng.choice(np.arange(self.p.offsets[anchor],self.p.offsets[anchor+1]),g,replace=False)))
            if value not in seen: positive.append(value); seen.add(value)
            if len(positive)==min(16,math.comb(count,g)): break
        if len(positive)!=min(16,math.comb(count,g)): raise ValueError('Positive group capacity exhausted')
        negative=[]; used=Counter(); expansions=[]; kin_radii=[]
        # F uses a bounded parent-first reservoir. Membership is still intact;
        # select closest predeclared descriptor tier, then parent then group.
        proposals=None
        for law,n in zip('GSKF',quotas(128-len(positive))):
            for _ in range(n):
                if law=='F':
                    if proposals is None:
                        prng=self.rng(epoch,anchor,'group_F_proposals',gmax)
                        pools={}
                        for c in range(11):
                            eligible=self.classes[c]
                            eligible=eligible[(self.sizes[eligible]>=g)&~np.isin(eligible,list(self.excluded(anchor)))]
                            if len(eligible): pools[c]=eligible
                        if not pools: raise ValueError('Group proposal classes lack capacity')
                        parents=np.empty(GROUP_PROPOSALS,np.int64); classes=prng.choice(list(pools),GROUP_PROPOSALS)
                        for c,eligible in pools.items(): parents[classes==c]=prng.choice(eligible,np.count_nonzero(classes==c))
                        # Uniform unordered g-subsets by rejection of repeated local slots.
                        local=(prng.random((GROUP_PROPOSALS,g))*self.sizes[parents,None]).astype(np.int64)
                        duplicate=(np.diff(np.sort(local,axis=-1),axis=-1)==0).any(-1)
                        while duplicate.any():
                            local[duplicate]=(prng.random((int(duplicate.sum()),g))*self.sizes[parents[duplicate],None]).astype(np.int64)
                            duplicate=(np.diff(np.sort(local,axis=-1),axis=-1)==0).any(-1)
                        proposal_array=np.unique(np.sort(local+self.p.offsets[parents,None],axis=-1),axis=0)
                        proposals=proposal_array
                        pd=self.d[proposal_array]; pn=self.n[proposal_array]
                        po=np.lexsort(tuple(np.concatenate((pd,pn),-1)[...,j] for j in reversed(range(5))),axis=-1)
                        pd=np.take_along_axis(pd,po[...,None],axis=1).reshape(len(proposals),-1)
                        pn=np.take_along_axis(pn,po[...,None],axis=1).reshape(len(proposals),-1)
                        proposal_parents=self.p.parents[proposal_array[:,0]]
                        codes=proposal_array.view(np.dtype((np.void,g*8))).reshape(-1)
                        seen_codes=np.asarray(list(seen),np.int64).view(np.dtype((np.void,g*8))).reshape(-1)
                        live=~np.isin(codes,seen_codes)
                        for p,n_used in used.items():
                            if n_used>=8: live[proposal_parents==p]=False
                    target=positive[int(rng.integers(len(positive)))]; td,tn=self.group_signature(target)
                    eligible=live
                    if not eligible.any(): raise ValueError('Group F reservoir exhausted; no quota relaxation')
                    dd=np.count_nonzero(pd!=td,axis=-1); nd=np.abs(pn-tn).max(-1)
                    discrete=dd[eligible].min(); radius=nd[eligible&(dd==discrete)].min()
                    pool=np.flatnonzero(eligible&(dd==discrete)&(nd==radius))
                    parents=self.p.parents[proposal_array[pool,0]]; parent=int(rng.choice(np.unique(parents)))
                    selected_proposal=int(pool[rng.choice(np.flatnonzero(parents==parent))])
                    value=tuple(proposals[selected_proposal]); expansions.append((int(discrete),int(radius)))
                else:
                    for _ in range(10000):
                        parent=self.parent(anchor,law,rng,used,8,g)
                        if law=='K': kin_radii.append(self.last_kinematic_radius)
                        if math.comb(int(self.p.offsets[parent+1]-self.p.offsets[parent]),g)<=used[parent]:
                            used[parent]=8; continue
                        value=tuple(sorted(rng.choice(np.arange(self.p.offsets[parent],self.p.offsets[parent+1]),g,replace=False)))
                        if value not in seen: break
                    else: raise ValueError('Group uniqueness capacity exhausted')
                negative.append(value); seen.add(value); used[int(self.p.parents[value[0]])]+=1
                if proposals is not None:
                    live[selected_proposal]=False
                    p=int(self.p.parents[value[0]])
                    if used[p]>=8: live[proposal_parents==p]=False
        indices=np.array(positive+negative,np.int64); labels=np.arange(128)<len(positive)
        if len(seen)!=128 or any(len(set(self.p.parents[v]))!=1 for v in indices): raise ValueError('Non-intact or duplicate groups')
        permutation=self.rng(epoch,anchor,'group_shuffle',gmax).permutation(128)
        return dict(indices=indices[permutation],labels=labels[permutation],evidence=dict(
            positives=len(positive),members=g,negative_parents=len(used),feature_proposals=GROUP_PROPOSALS,
            max_kinematic_radius=max(kin_radii,default=0),max_feature_tier=list(max(expansions,default=(0,0)))))

    def group_signature(self,indices):
        # Member ordering is canonical by discrete descriptor then numeric bins.
        d=self.d[list(indices)]; n=self.n[list(indices)]
        order=np.lexsort(tuple(np.column_stack((d,n))[:,j] for j in reversed(range(5))))
        return d[order].flatten(),n[order].flatten()

    def batch(self,anchors,epoch,cfg,mine=None):
        rows=[]
        for j,i in enumerate(anchors):
            if cfg['family']=='P': rows.append(self.particle_pool(int(i),epoch,cfg['law'],None if mine is None else lambda f:mine(j,f)))
            else: rows.append(self.group_pool(int(i),epoch,cfg['group_size']))
        evidence=[r.pop('evidence') for r in rows]
        labels=np.stack([r['labels'] for r in rows])
        if cfg['family']=='P': features=self.p.features[np.stack([r['indices'] for r in rows])]; result=dict(features=features,labels=labels)
        else:
            indices=np.zeros((len(rows),128,cfg['group_size']),np.int64); mask=np.zeros(indices.shape,bool)
            for j,r in enumerate(rows):
                g=r['indices'].shape[1]; indices[j,:,:g]=r['indices']; mask[j,:,:g]=True
            local=np.where(mask[...,None],self.p.local[indices],0.)
            features=np.where(mask[...,None],self.p.features[indices],0.)
            pairs,_=pair_features(local,mask)
            result=dict(features=features,labels=labels,member_mask=mask,relations=pairs)
        return result,evidence


class JetCandidates:
    def __init__(self,offline_keys,identities,labels,pt,count,*,source_keys=None,replicate=0,device='cpu'):
        # Even J6 mines on original OFFLINE keys: only the scored key source changes.
        self.sampler=JetSampler(offline_keys,identities,labels,pt=pt,count=count,root_seed=2026103102+replicate,replicate=replicate,device=device)
        self.capacity=self.sampler.capacity
        self.keys=offline_keys if source_keys is None else source_keys
        if self.keys.shape!=offline_keys.shape or self.keys.dtype!=np.float32 or not np.isfinite(self.keys).all():
            raise ValueError('Invalid scored jet keys')

    def batch(self,anchors,epoch,cfg,mine=None):
        ix,info=self.sampler.sample(anchors,epoch,'mix')
        return dict(features=self.keys[ix],positive=np.zeros(len(anchors),np.int64)),info


def tensors(batch,device):
    return {k:torch.as_tensor(v,device=device).detach() for k,v in batch.items()}
