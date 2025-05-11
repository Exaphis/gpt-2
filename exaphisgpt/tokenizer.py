import json
from pathlib import Path
import regex as re
from typing import Generator


# Overview of the HF tokenization pipeline: https://huggingface.co/docs/tokenizers/en/pipeline
# Reference for BPE tokenization: https://huggingface.co/learn/llm-course/en/chapter6/5#implementing-bpe
class Tokenizer:
    """GPT-2 byte pair encoding (BPE) tokenizer."""

    def __init__(self, config_file: str):
        with open(config_file, "r") as f:
            self.config = json.load(f)

        self.merges: dict[tuple[str, str], str] = {}
        for s in self.config["model"]["merges"]:
            a, b = s.split(" ")
            self.merges[a, b] = a + b

        self.vocab: dict[str, int] = self.config["model"]["vocab"]

        # Regex to preprocess the text into words
        # Taken from https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py#L53
        # This needs the regex module to support "\p{L}" and "\p{N}" (matches any unicode letter or number)
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        # Set up transformation rules for UTF-8 bytes to unicode strings.
        # This:
        # 1. Converts whitespace/control characters to readable unicode characters (e.g., space -> Ġ).
        # 2. Avoids the vocab having to contain all possible unicode characters.
        # Taken from https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py#L9
        printable = (
            list(range(ord("!"), ord("~") + 1))  # ASCII printable: 33–126
            + list(range(ord("¡"), ord("¬") + 1))  # Latin-1 Supplement: 161–172
            + list(range(ord("®"), ord("ÿ") + 1))  # Latin-1 Supplement: 174–255
        )
        self.byte_map: dict[int, str] = {c: chr(c) for c in printable}
        unseen = 0
        for i in range(256):
            if i not in self.byte_map:
                self.byte_map[i] = chr(unseen + 256)  # Map to a new unicode character
                unseen += 1

    def pretokenize(self, text: str) -> Generator[str, None, None]:
        """Pre-tokenize the input text into words and mperform the byte-level encoding."""
        for match in self.pat.finditer(text):
            match_str = match.group(0)
            yield "".join(self.byte_map[b] for b in match_str.encode("utf-8"))

    def encode(self, text: str) -> list[int]:
        """Encode the input text into a list of token IDs."""

        result = []
        for word in self.pretokenize(text):
            # We can't greedily tokenize the word by choosing the longest match in the vocab,
            # since this doesn't take into account different merge priorities.
            # Instead, we need to iterate over each merge rule and apply it to the word.
            chars = list(word)
            for pair, merge_result in self.merges.items():
                new_chars = []
                for c in chars:
                    new_chars.append(c)
                    if len(new_chars) > 1 and (new_chars[-2], new_chars[-1]) == pair:
                        # Merge the last two characters
                        new_chars[-2:] = [merge_result]

                chars = new_chars
            result.extend(chars)

        return [self.vocab[token] for token in result]


if __name__ == "__main__":
    from transformers import AutoTokenizer
    import argparse

    my_tokenizer = Tokenizer(Path(__file__).parent / "../gpt2/tokenizer.json")
    hf_tokenizer = AutoTokenizer.from_pretrained(Path(__file__).parent / "../gpt2")

    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str, help="Text to tokenize")

    args = parser.parse_args()
    text = args.text

    print(f"Text: {text}")
    hf_res = hf_tokenizer.encode(text)
    print(f"HuggingFace: {hf_res}")
    my_res = my_tokenizer.encode(text)
    print(f"exaphisgpt:  {my_res}")
