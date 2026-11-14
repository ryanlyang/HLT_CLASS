"""Record local split producer bytes separately from the relocatable foundation."""
from pathlib import Path
import subprocess
from hlt_classification.data.cache_contracts import load_json,sha256_file,write_immutable_json
from . import contracts as c

FILES=('src/hlt_classification/luka_new_fullstat_hydra/population.py',
       'src/hlt_classification/luka_new_fullstat_hydra/contracts.py',
       'src/hlt_classification/luka_fullstat/data.py',
       'src/hlt_classification/jetclass2_delphes/schema.py',
       'src/hlt_classification/jetclass2_delphes/inventory.py',
       'src/hlt_classification/data/cache_contracts.py')


def record(dataset,project):
    dataset,project=Path(dataset),Path(project)
    f=load_json(dataset/'foundation/foundation.json');c.validate(f,'FOUNDATION')
    result=c.artifact('FOUNDATION_PRODUCER',foundation_sha256=f['content_hash'],
        base_commit=subprocess.run(['git','-C',str(project),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip(),
        producer_files={name:sha256_file(project/name) for name in FILES},
        local_source_clean_claim=False,description='Local scalar split; actual generator bytes pinned separately from base commit')
    write_immutable_json(dataset/'foundation/producer.json',result)
    return result


def verify(dataset,project,foundation_hash):
    p=load_json(Path(dataset)/'foundation/producer.json');c.validate(p,'FOUNDATION_PRODUCER')
    if p['foundation_sha256']!=foundation_hash or p['producer_files']!={name:sha256_file(Path(project)/name) for name in FILES}:
        raise ValueError('Split producer semantics differ from pinned execution source')
    return p
