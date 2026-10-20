"""Original ragged FP32 target schema, authenticated once into bounded RAM."""
from hashlib import sha256
from io import BytesIO
import numpy as np
from hlt_classification.cms_proxy_ladder.contracts import safe
from hlt_classification.luka_attention.banks import Bank, check_matrices


class ResidentBank(Bank):
    def __init__(self, *args, max_bytes, **kwargs):
        super().__init__(*args, **kwargs)
        elements = sum(int(n)**2 for n in self.counts) if self.manifest['attention'] else 0
        self.loaded_bytes = elements*4+(len(self.counts)+1)*8+self.probabilities.nbytes+self.identities.nbytes+self.counts.nbytes
        if type(max_bytes) is not int or self.loaded_bytes > max_bytes:
            raise MemoryError('Resident teacher bank exceeds registered RAM budget')
        self.offsets = np.concatenate(([0], np.cumsum(self.counts**2))).astype(np.int64)
        self.values = np.empty(elements, np.float32)
        for shard in self.manifest['shards']:
            ref = shard['payload']
            blob = safe(self.root, ref['path']).read_bytes()
            if len(blob) != ref['bytes'] or sha256(blob).hexdigest() != ref['sha256']:
                raise ValueError('Teacher payload changed during resident copy')
            values = np.load(BytesIO(blob), allow_pickle=False)
            a,b = shard['start'],shard['stop']
            check_matrices(values, np.asarray(shard['offsets'], np.int64), self.counts[a:b])
            self.values[self.offsets[a]:self.offsets[b]] = values
        for a in (self.values,self.offsets,self.counts,self.probabilities,self.identities):
            a.flags.writeable = False

    def attention_batch(self, indices, width):
        ix = np.asarray(indices)
        if (not self.manifest['attention'] or ix.ndim != 1 or ix.dtype.kind not in 'iu' or not len(ix)
                or np.any(ix < 0) or np.any(ix >= len(self.counts))
                or not isinstance(width, (int, np.integer)) or not 1 <= width <= 512
                or width < self.counts[ix].max()):
            raise ValueError('Invalid resident attention selection')
        out = np.zeros((len(ix),width,width), np.float32)
        for dest, i in enumerate(ix):
            n = self.counts[i]
            out[dest,:n,:n] = self.values[self.offsets[i]:self.offsets[i+1]].reshape(n,n)
        return out
