"""Same authenticated HLT/U100/T00 population, new execution-bound subclass bank."""
from hlt_classification.luka_subclass_hydra.inputs import Bundle, validate_sources, offline_bank, join
from hlt_classification.luka_subclass_hydra import inputs as donor
from .contracts import cell
from . import targets


def views(spec,task):
    cell(task)
    return donor.views(spec,'S0p5')


def banks(spec,task,train):
    cell(task)
    return dict(u100=donor.embedding_bank(spec,train),subclass=targets.Bank(spec,train))


def fit_parents(spec,task):
    cell(task)
    return dict(spec=spec['content_hash'],recipe=spec['recipe']['content_hash'],u100=spec['embedding_hash'],
                subclass=targets.check_manifest(spec)['content_hash'])
