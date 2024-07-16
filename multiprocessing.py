#
# Copyright 2020-2021 Enflame. All Rights Reserved.
#
from __future__ import division
from __future__ import print_function

import sys
import torch.multiprocessing
import traceback

def _mp_start_fn(index, fn, args):
    exit_code = 0
    try:
        fn(index, *args)
    except Exception as e:
        print(
            'Exception in device=xla:{}  {}'.format(index, str(e)),
            file=sys.stderr)
        traceback.print_exc(limit=16, file=sys.stderr)
        exit_code = 17
    sys.exit(exit_code)


def spawn(fn,
          args=(),
          nprocs=None,
          join=True,
          daemon=False,
          start_method='spawn'):
    """Enables multi processing based replication.

    Args:
      fn (callable): The function to be called for each device which takes part of
        the replication. The function will be called with a first argument being
        the global index of the process within the replication, followed by the
        arguments passed in `args`.
      args (tuple): The arguments for `fn`.
        Default: Empty tuple
      nprocs (int): The number of processes/devices for the replication. At the
        moment, if specified, can be either 1 or the maximum number of devices.
      join (bool): Whether the call should block waiting for the completion of the
        processes which have being spawned.
        Default: True
      daemon (bool): Whether the processes being spawned should have the `daemon`
        flag set (see Python multi-processing API).
        Default: False
      start_method (string): The Python `multiprocessing` process creation method.
        Default: `spawn`

    Returns:
      The same object returned by the `torch.multiprocessing.spawn` API. If
      `nprocs` is 1 the `fn` function will be called directly, and the API will
      not return.
    """

    return torch.multiprocessing.start_processes(
            _mp_start_fn,
            args=(fn, args),
            nprocs=nprocs,
            join=join,
            daemon=daemon,
            start_method=start_method)

class MpSerialExecutor(object):
    """Utility to run a function in a serialized fashion among multi-core processes.

    Example::

      # At global scope.
      SERIAL_EXEC = xmp.MpSerialExecutor()

      def load_dataset(path):
        return maybe_download_and_load(path)

      def _mp_fn(index, ...):
        # Avoid all cores downloading the same data with the serial executor.
        dataset = SERIAL_EXEC.run(lambda: load_dataset('/tmp/mnist-data'))
        ...

      xmp.spawn(_mp_fn, ...)
    """

    def __init__(self):
        self._lock = torch.multiprocessing.Lock()

    def run(self, fn):
        """Runs the provided function serialized WRT each per-core process.

        Args:
          fn (callable): The function to run in a serialized fashion.
        Returns:
          The `fn` return value.
        """
        with self._lock:
            return fn()