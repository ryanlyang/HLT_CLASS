"""Flushed execution stages and stack samples; never scientific artifacts."""
from contextlib import contextmanager
import atexit
import faulthandler
import os
import time

_INTERVAL = None


def enable_stall_traces(seconds=300):
    """Show the active stack after a gap in progress, without aborting work."""
    if type(seconds) is not int or seconds <= 0:
        raise ValueError('Positive integer stall interval required')
    global _INTERVAL
    if _INTERVAL is None:
        faulthandler.enable()
        atexit.register(faulthandler.cancel_dump_traceback_later)
    _INTERVAL = seconds
    mark('diagnostics enabled; a scheduled stack sample is not a failed check')


def mark(message):
    if _INTERVAL is not None:
        faulthandler.dump_traceback_later(_INTERVAL, repeat=True)
    rank = os.environ.get('RANK', 'launcher')
    print(f'LUKABREC_STAGE pid={os.getpid()} rank={rank} {message}', flush=True)


def disable_stall_traces():
    """A parent waiting on workers and ordinary science epochs are not stalls."""
    global _INTERVAL
    if _INTERVAL is not None:
        faulthandler.cancel_dump_traceback_later()
        _INTERVAL = None


@contextmanager
def stage(name):
    begun = time.monotonic()
    mark('BEGIN ' + name)
    try:
        yield
    except BaseException as error:
        mark(f'FAILED {name} elapsed_s={time.monotonic()-begun:.1f} error={type(error).__name__}')
        raise
    else:
        mark(f'END {name} elapsed_s={time.monotonic()-begun:.1f}')
