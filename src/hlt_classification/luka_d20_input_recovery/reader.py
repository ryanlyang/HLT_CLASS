"""Native local handle; retain the original basket/interpretation implementation.

Constructor adapted from luka_d20_full_hydra/basket_reader.py at 5af66449.
"""
import uproot


def make_reader(base):
    class NativeReader(base):
        def __init__(self,path,tree_key,*,cache_bytes=128*2**20):
            from collections import OrderedDict
            if type(cache_bytes) is not int or cache_bytes<0:raise ValueError('Invalid basket-cache budget')
            self.handle=uproot.open(path,handler=uproot.source.file.MultithreadedFileSource,
                num_workers=1,use_threads=False,object_cache=None,array_cache=None)
            try:
                if not isinstance(self.handle.file.source,uproot.source.file.MultithreadedFileSource):
                    raise ValueError('Persistent native local source required')
                self.tree=self.handle[tree_key]
            except BaseException:
                self.handle.close();raise
            self.cache_bytes=cache_bytes;self.cache=OrderedDict()
            self.resident_bytes=self.peak_bytes=self.reads=self.hits=0
            self.library=uproot.interpretation.library.Awkward()
            self.numpy_library=uproot.interpretation.library.NumPy()
            self.num_entries=self.tree.num_entries
    return NativeReader
