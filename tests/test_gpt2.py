from pathlib import Path
from exaphisgpt.gpt2 import read_safetensors
from safetensors import safe_open
import numpy as np


def test_read_safetensors():
    # Test the read_safetensors function
    model_dir = Path(__file__).parent / "../gpt2"
    tensors = read_safetensors(model_dir / "model.safetensors")
    with safe_open(
        model_dir / "model.safetensors", framework="numpy", device="cpu"
    ) as f:
        for name in f.keys():
            tensor = np.array(tensors[name], dtype="float32")
            hf_tensor = f.get_tensor(name)
            assert np.array_equal(tensor, hf_tensor), f"Tensor {name} does not match"
