import json
from pathlib import Path
import typing
import copy
import struct
import functools
import itertools
import tqdm


def load_tensor(b: bytes, dtype: str, shape: list[int]):
    """
    Read a tensor from a safetensors file.

    Args:
        bytes: The bytes object containing the tensor data.
        dtype (str): The data type of the tensor.
        shape (list[int]): The shape of the tensor.

    Returns:
        list[float]: A len(shape)-dimensional list of floats representing the tensor.
    """
    if dtype != "F32":
        raise ValueError(f"Unsupported dtype: {dtype}")

    num_elems = functools.reduce(lambda x, y: x * y, shape)

    tensor = 0
    for dim in reversed(shape):
        tensor = [copy.copy(tensor) for _ in range(dim)]

    # Make sure the tensor is of the right shape
    tmp_tensor = tensor
    for dim in shape:
        assert len(tmp_tensor) == dim
        tmp_tensor = tmp_tensor[0]
    del tmp_tensor

    idxs = itertools.product(*[range(dim) for dim in shape])
    for idx, val in zip(idxs, struct.unpack(f"<{num_elems}f", b)):
        tmp_tensor = tensor
        for i in idx[:-1]:
            tmp_tensor = tmp_tensor[i]
        tmp_tensor[idx[-1]] = val

    return tensor


def read_safetensors(filename: str) -> dict[str, list]:
    # parse safetensors
    # https://huggingface.co/docs/safetensors/en/index
    tensors = {}
    with open(filename, "rb") as f:
        # 1. 8 bytes representing the size of the header
        header_size = int.from_bytes(f.read(8), "little")
        # 2. header size bytes containing the header encoded as a UTF-8 string
        header = json.loads(f.read(header_size).decode("utf-8"))

        for name in (pbar := tqdm.tqdm(header, desc="Loading tensors")):
            pbar.set_description_str(f"Loading tensors ({name:<25})")
            if name == "__metadata__":
                continue

            dtype = header[name]["dtype"]
            shape = header[name]["shape"]
            data_offsets = header[name]["data_offsets"]
            # Seek to correct offset (note: data_offset is from the start of the byte buffer,
            # not the start of the file)
            f.seek(8 + header_size + data_offsets[0])
            data = f.read(data_offsets[1] - data_offsets[0])
            tensors[name] = load_tensor(data, dtype, shape)

    return tensors


class GPT2:
    def __init__(self, model_dir: Path):
        with open(model_dir / "config.json", "r") as f:
            config = json.load(f)
        self.vocab_size = config["vocab_size"]
        self.n_ctx = config["n_ctx"]
        self.n_embd = config["n_embd"]
        self.n_head = config["n_head"]
        self.n_layer = config["n_layer"]
        self.tensors = read_safetensors(model_dir / "model.safetensors")
        print(self.tensors.keys())

    def infer(inputs: list[int]) -> list[list[float]]:
        """
        GPT-2 inference.

        Args:
            inputs (list[int]): List of token IDs.

        Returns:
            list[list[float]]: For each position in the sequence, a list of predicted token probabilities.
        """


if __name__ == "__main__":
    model_dir = Path(__file__).parent / "../gpt2"
    gpt2 = GPT2(model_dir)
    print(gpt2.vocab_size)
    print(gpt2.n_ctx)
    print(gpt2.n_embd)
    print(gpt2.n_head)
    print(gpt2.n_layer)
