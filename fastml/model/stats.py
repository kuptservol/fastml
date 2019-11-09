from fastml.core import *
from fastml.data import *
from fastml.examples import examples
from fastml.model.model import *
from fastml.model.image.cnn import *

class Hook():
    def __init__(self, layer, hook_func): self.hook = layer.register_forward_hook(partial(hook_func, self))
    def remove(self): self.hook.remove()
    def __del__(self): self.remove()

def append_stats(hook, model, inp, outp):
    if not hasattr(hook,'stats'): hook.stats = ([],[])
    means,stds = hook.stats
    means.append(outp.data.mean())
    stds.append(outp.data.std())

def children(m): return list(m.children())

class Hooks(ListContainer):
    def __init__(self, layers, hook_func): super().__init__([Hook(layer, hook_func) for layer in layers])
    def __enter__(self, *args): return self
    def __exit__ (self, *args): self.remove()
    def __del__(self): self.remove()

    def __delitem__(self, i):
        self[i].remove()
        super().__delitem__(i)

    def remove(self):
        for h in self: h.remove()