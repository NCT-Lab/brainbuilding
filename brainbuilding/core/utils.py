import numpy as np
from pathlib import Path
import importlib.resources


def resolve_resource_path(relative_path: str) -> str:
    """
    Finds a resource file's full path, searching first relative to CWD
    for development, then inside the package for installed distribution.
    """
    dev_path = Path(f"{relative_path}")
    if dev_path.exists():
        return dev_path.as_posix()

    try:
        with importlib.resources.as_file(
            importlib.resources.files("brainbuilding").joinpath(relative_path)
        ) as p:
            if p.exists():
                return p.as_posix()
    except (FileNotFoundError, ModuleNotFoundError):
        pass

    raise FileNotFoundError(
        f"Resource not found: '{relative_path}'. Looked for dev path "
        f"'{dev_path}' and as a package resource."
    )


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
