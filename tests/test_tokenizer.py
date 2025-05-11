from transformers import AutoTokenizer
from exaphisgpt.tokenizer import Tokenizer
from hypothesis import given, settings, example, strategies as st


@given(st.text())
@example("\xad")
@settings(deadline=1000)
def test_tokenizer_1(text):
    hf_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer = Tokenizer("gpt2/tokenizer.json")
    assert tokenizer.encode(text) == hf_tokenizer.encode(text)
