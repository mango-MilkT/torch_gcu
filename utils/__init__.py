#
# Copyright 2019-2021 Enflame. All Rights Reserved.
#
class LazyProperty(object):

    def __init__(self, gen_fn):
        self._gen_fn = gen_fn

    @property
    def value(self):
        if self._gen_fn is not None:
            self._value = self._gen_fn()
            self._gen_fn = None
        return self._value
