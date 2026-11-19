import numpy as np


def extrema_indices(series, max_points):
    arrays = [np.asarray(values) for values in series]
    length = len(arrays[0])
    if length <= max_points:
        return np.arange(length, dtype=np.int64)
    important = {0, length - 1}
    budget_per_bin = 2 * len(arrays) + 2
    bins = max(1, (max_points - 2) // budget_per_bin)
    boundaries = np.linspace(0, length, bins + 1, dtype=np.int64)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        important.update((int(start), int(stop - 1)))
        for values in arrays:
            local = values[start:stop]
            finite_index = np.flatnonzero(np.isfinite(local))
            if not len(finite_index):
                continue
            finite = local[finite_index]
            important.add(int(start + finite_index[np.argmin(finite)]))
            important.add(int(start + finite_index[np.argmax(finite)]))
    ordered = np.array(sorted(important), dtype=np.int64)
    if len(ordered) > max_points:
        return ordered[np.linspace(0, len(ordered) - 1, max_points, dtype=np.int64)]
    return ordered

