import h5py
from typing import Optional, Any, List, Dict, Tuple, Union, Callable, TypeVar, Mapping, Sequence
import numpy as np

LeafType = TypeVar("LeafType")
TensorMapping = Union[Mapping[Any,"TensorMapping[LeafType]"], LeafType]

T = TypeVar('T')
def _flatten_tensor_tree(src_tree : TensorMapping[T]) -> dict[tuple,T]:
    """Flattens a tensor tree, returning a tensor tree with not subtrees,
    defined as a dictionary, with tuples as keys.

    Parameters
    ----------
    src_tree : TensorTree
        Tensor tree to ble flattened

    Returns
    -------
    dict
        _description_
    """
    if isinstance(src_tree, dict):
        r = {}
        for k in src_tree.keys():
            flat_subtree = _flatten_tensor_tree(src_tree[k])
            for sk,sv in flat_subtree.items():
                r[(k,)+sk] = sv
        return r
    else:
        return {tuple():src_tree}



def to_string_array(strings : list[str] | np.ndarray, max_string_len : int = 32):
    return np.array([list(n.encode("utf-8").ljust(max_string_len)[:max_string_len]) for n in strings], dtype=np.uint8) # ugly, but simple

def dump(filename : str, data : TensorMapping[np.ndarray], labels : TensorMapping[np.ndarray | None] | None = None):
    """Dumps the data and labels to an HDF5 file.

    Parameters
    ----------
    filename : str
        Path to the output HDF5 file.
    data : dict
        Data to be saved, must be a nested dict of numpy arrays.
    labels : dict | None
        Labels to be saved, must be a nested dict of numpy arrays. The numpy arrays must contain labels for each column of the corresponding data.
        You con generate a proper labels dict by using the `to_string_array` function on a list of strings.
    """
    data = _flatten_tensor_tree(data)
    if labels is not None:
        labels = _flatten_tensor_tree(labels)
        labels = {k:np.expand_dims(v, axis=0) if v is not None else None for k,v in labels.items()}

    with h5py.File(filename, "w") as f:
        for k,v in data.items():
            field_name = ".".join(k)
            f.create_dataset(field_name, data=v)
            if labels is not None and k in labels and labels[k] is not None:
                f.create_dataset(field_name+"_labels", data=labels[k])


