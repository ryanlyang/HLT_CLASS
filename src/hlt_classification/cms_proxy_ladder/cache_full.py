"""Full-campaign preparation policy; old foundations retain their own policy."""
from pathlib import Path

import numpy as np


def ordered_results(pool, function, arguments, workers):
    """Bound submissions and completed-but-unconsumed results, preserving order."""
    iterator = iter(arguments)
    pending = [pool.submit(function, arg)
               for arg in (next(iterator, None) for _ in range(workers)) if arg is not None]
    while pending:
        yield pending.pop(0).result()
        argument = next(iterator, None)
        if argument is not None:
            pending.append(pool.submit(function, argument))


def preparation_bound(foundation, role, workers):
    from .release import load_bank
    if role not in ("train", "validation"):
        raise PermissionError("Full context cache cannot access final test")
    if type(workers) is not int or not 1 <= workers <= 72:
        raise ValueError("Invalid full context preparation worker count")
    if foundation["cache_preparation"] != "source_file_bounded/v1":
        raise ValueError("Full context cache policy differs")
    arrays = load_bank(foundation["release"], root=Path(foundation["release_root"]))
    code = 0 if role == "train" else 1
    counts = np.bincount(arrays["source_file"][arrays["role"] == code],
                         minlength=len(foundation["release"]["source_files"]))
    rows = foundation["role_counts"][role]
    if int(counts.sum()) != rows or not rows:
        raise ValueError("Full context cache source population differs")
    active = min(workers, int(np.count_nonzero(counts)))
    # Same conservative capacity charge as legacy code; no mean-count estimate.
    per_row = foundation["inputs"]["capacity"] * 21 * 4 + 128
    resident = rows * per_row
    # Lists, concatenation and IPC copies for the largest active single files.
    in_flight = 4 * int(np.sort(counts)[-active:].sum()) * (per_row + 1024)
    # Workers load full assignment banks, role copies, release indices and
    # temporary identity validation sets. Charge these separately from p4/views.
    all_rows = sum(foundation["role_counts"].values())
    bookkeeping = active * (8 * foundation["assignment_slots"] + 256 * all_rows + 512 * 2**20)
    return resident + in_flight + bookkeeping
