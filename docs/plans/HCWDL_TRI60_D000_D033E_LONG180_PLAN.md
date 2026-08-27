# TRI60 D000-from-D033E 180-pass study

Status: implementation-authoritative additive validation study.

## Question

The original full-data LOGIT D000 specialists selected these checkpoints:

```text
D000 <- U000    50/60   AUC 0.946170
D000 <- U050E   51/60   AUC 0.946213
D000 <- U100E   60/60   AUC 0.946254
D000 <- D066E   60/60   AUC 0.946606
D000 <- D033E   60/60   AUC 0.946764
```

The nearest teacher produced the strongest specialist while selecting the
last available pass. This study asks whether a longer optimization horizon
improves that exact edge.

## Frozen comparison

One fresh full-data fit reproduces the original
`LOGIT_D000_from_D033E` edge except for its registered 180-pass horizon:

```text
teacher                  LOGIT_D033E probability ensemble
student view             byte-exact canonical HLT (D000)
architecture             unified 21-channel Particle Transformer
loss                     0.25 CE + 0.75 forward probability KD
temperature              2.0
initialization/seed       same alias as original D000 <- D033E specialist
batch size               256
optimizer                AdamW, peak LR 3e-4, weight decay 0.01
schedule                 5% warmup, cosine decay, 5% floor over 180 passes
validation               every pass
selection                macro AUC, CE, logR50, earliest update
```

Temperature 2 is essential: temperature 1 belongs to the M1 compression edge,
not the original D000 specialist.

The 180-pass cosine schedule has a different horizon from the original
60-pass schedule. Therefore pass 60 of the new run is reported but is not
claimed to reproduce the original pass-60 optimizer state. The compact
comparison stores end-of-pass metrics at 60, 120, and 180, the globally
selected checkpoint, and the original 60-pass selected result.

## Isolation and resources

The study consumes the authenticated existing `LOGIT_D033E` train probability
bank read-only. It neither copies nor republishes that bank and has no
scheduler dependency on or write path into the original TRI60, DX, RSET,
RREL, M1-screen, or greedy-ensemble campaigns.

The single GPU job requests 72 CPUs, 320 GiB RAM, one GH200, and 72 hours. The
full exact-HLT train and validation views exist only in process RAM. Durable
outputs are the selected/final checkpoints, the normal 180-pass report, one
compact comparison report, logs, and task attestation. Rolling resumes and
partial checkpoint reuse are forbidden; an interruption restarts from zero.

No standalone smoke is required. This additive run reuses the authenticated
production worker, full-data foundation, exact-HLT cache, and teacher-bank
evidence. It remains validation-only and cannot access final test or choose a
campaign finalist. Finite poor performance completes normally.
