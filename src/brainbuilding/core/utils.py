import numpy as np


def happend(x, col_data, col_name: str):
    if not x.dtype.fields:
        # Not a structured array
        return None
    # 0) create new structured array
    old_dtype = [i for i in x.dtype.descr if i[0] != col_name and i[0] != ""]
    y = np.empty(
        x.shape,
        dtype=old_dtype + [(col_name, col_data.dtype, col_data.shape[1:])],
    )
    for name in [i[0] for i in old_dtype]:
        # 1) copy old array
        y[name] = x[name]

    y[col_name] = col_data
    return y