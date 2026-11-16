"""Registration-time serialized inventory; setup replaces projections with measurements."""
import io
import math
import torch
from .contracts import artifact,READOUTS,FITS,QUALIFICATION_TASKS
from .optimization import isolated_rng


def serialized_bytes(state):
    stream=io.BytesIO();torch.save(state,stream);return stream.tell()


def inventory(maximum,imported,counts):
    from .kernel import build
    from .resources import POSITIONS
    with isolated_rng('cpu',seed=2026111505):
        model,_=build(dict(maximum=maximum,imported=imported),'R24_s7',0,1,'cpu')
        native=serialized_bytes(model.deployment_state());full=serialized_bytes(model.state_dict())
        parameters=sum(p.numel()*p.element_size() for p in model.parameters());del model
    predictions=len(READOUTS)*(counts['validation']*(32+8+11*4)+1024**2)
    # AdamW two moments plus per-parameter scalar/serialization overhead. The
    # measured full state already includes decoder buffers. Both EMA copies
    # contain native parameters only; native serialization overbounds them.
    final_state=full+2*parameters+2*native+4*1024**2
    startup=final_state+len(READOUTS)*(native+1024**2)+8*1024**2
    fit=final_state+len(READOUTS)*native+predictions+startup+32*1024**2
    # Complete common-start references are broadcast in memory. Saved named
    # comparisons/hashes are small; bound their JSON and logging at 2 MiB each.
    parity=len(QUALIFICATION_TASKS)*len(POSITIONS)*2*1024**2
    setup=len(FITS)*startup+parity+2*1024**3
    required=math.ceil((1.25*(len(FITS)*fit+setup)+2**30)/2**30)*2**30
    return artifact('STORAGE_INVENTORY',maximum=maximum,validation_rows=counts['validation'],
        native_serialized_bytes=native,full_serialized_bytes=full,parameter_bytes=parameters,
        estimated_fit_bytes=fit,estimated_setup_bytes=setup,output_budget_bytes=required,
        safety_multiplier=1.25,readouts=list(READOUTS),fits=len(FITS),reference_records=len(QUALIFICATION_TASKS)*len(POSITIONS))


def validate_inventory(value,spec):
    from .contracts import validate
    validate(value,'STORAGE_INVENTORY')
    if value!=inventory(spec['maximum'],spec['imported'],spec['counts']):raise ValueError('Storage inventory changed')
    if spec['output_budget_bytes']!=value['output_budget_bytes'] or spec['quota_gib']*2**30<spec['output_budget_bytes']+2**30:
        raise ValueError('Personally available output quota is below the measured inventory plus reserve')
