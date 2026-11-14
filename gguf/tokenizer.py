"""Rebuilds a Hugging Face tokenizer from the vocabulary and merges stored in a GGUF file."""
from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers

# Pre-tokenizer regexes, from the models' tokenizer.json (llama.cpp's tokenizer.ggml.pre names them)
PRE_TOKENIZER_REGEXES = {
    "qwen2": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    "qwen35": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
}
CONTROL_TOKEN, USER_DEFINED_TOKEN = 3, 4  # tokenizer.ggml.token_type values

def build_tokenizer(metadata: dict) -> Tokenizer:
    """Rebuilds the Hugging Face byte-level BPE tokenizer from the vocabulary and merges stored in the GGUF."""
    if metadata.get("tokenizer.ggml.model") != "gpt2":
        raise NotImplementedError(f"tokenizer type {metadata.get('tokenizer.ggml.model')!r} is not supported "
                                  "(only byte-level BPE, 'gpt2')")
    pre = metadata.get("tokenizer.ggml.pre")
    if pre not in PRE_TOKENIZER_REGEXES:
        raise NotImplementedError(f"pre-tokenizer {pre!r} is not supported (supported: {', '.join(PRE_TOKENIZER_REGEXES)})")

    tokens = metadata["tokenizer.ggml.tokens"]
    merges = [tuple(m.split(" ", 1)) for m in metadata["tokenizer.ggml.merges"]]
    tokenizer = Tokenizer(models.BPE({t: i for i, t in enumerate(tokens)}, merges))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PRE_TOKENIZER_REGEXES[pre]), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()

    # Control tokens (<|im_start|>, ...) and user-defined ones (<think>, ...) match as whole strings before BPE
    types = metadata.get("tokenizer.ggml.token_type", [])
    tokenizer.add_special_tokens([AddedToken(tokens[i], special=True, normalized=False)
                                  for i, kind in enumerate(types) if kind == CONTROL_TOKEN])
    tokenizer.add_tokens([AddedToken(tokens[i], special=False, normalized=False)
                          for i, kind in enumerate(types) if kind == USER_DEFINED_TOKEN])
    return tokenizer
