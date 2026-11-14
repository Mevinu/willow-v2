"""Loads Qwen3.8-Flash-Next (GGUF architecture qwen4exp) into model.py's engine. It shares Qwen3.6's GGUF naming
and converter fixes (engines/qwen36/loader.py) and adds the tensors only Qwen3.8 has. The n-gram (PLE) table is
never loaded: gguf.streamed.StreamedEmbedding reads the rows each token needs straight from the GGUF."""
from engines.qwen36.loader import (GLOBAL_NAMES, LAYER_NAMES, backbone_converter,
                                   build_config as build_qwen_config,
                                   build_engine, load_weights)
from gguf.streamed import StreamedEmbedding
from .model import Qwen38Engine

ARCHITECTURES = {"qwen4exp": Qwen38Engine}

PLE_TABLE = "per_layer_token_embd.weight"

# llama.cpp name -> Hugging Face name (transformers' qwen4_exp), relative to the layer
LAYER_NAMES_38 = {
    **LAYER_NAMES,
    "hc_attn_norm.weight": "attn_hyper_connection.hc_norm.weight",
    "hc_attn_down.weight": "attn_hyper_connection.input_mix_weight_down.weight",
    "hc_attn_up.weight": "attn_hyper_connection.input_mix_weight_up.weight",
    "hc_attn_inject.weight": "attn_hyper_connection.block_inject_weight.weight",
    "hc_ffn_norm.weight": "mlp_hyper_connection.hc_norm.weight",
    "hc_ffn_down.weight": "mlp_hyper_connection.input_mix_weight_down.weight",
    "hc_ffn_up.weight": "mlp_hyper_connection.input_mix_weight_up.weight",
    "hc_ffn_inject.weight": "mlp_hyper_connection.block_inject_weight.weight",
    "ple_key.weight": "ple.key_proj.weight",
    "ple_value.weight": "ple.value_proj.weight",
    "ple_norm_key.weight": "ple.norm_key.weight",
    "ple_norm_query.weight": "ple.norm_query.weight",
    "ple_norm_conv.weight": "ple.norm_conv.weight",
    "ple_conv1d.weight": "ple.conv1d.weight",  # [channels, kernel]: the converter dropped the middle dim
    # The QSA indexer (index_qk_proj, split in two by the converter). Loaded, not used yet: attention is dense
    "indexer.q_proj.weight": "self_attn.indexer.q_proj.weight",
    "indexer.k_proj.weight": "self_attn.indexer.k_proj.weight",
    "indexer.q_norm.weight": "self_attn.indexer.q_layernorm.weight",
    "indexer.k_norm.weight": "self_attn.indexer.k_layernorm.weight",
}
GLOBAL_NAMES_38 = {
    "token_embd.weight": GLOBAL_NAMES["token_embd.weight"],
    "output_hc_norm.weight": "hyper_connection_mixer.hc_norm.weight",
    "output_hc_down.weight": "hyper_connection_mixer.input_mix_weight_down.weight",
    "output_hc_up.weight": "hyper_connection_mixer.input_mix_weight_up.weight",
    PLE_TABLE: None,  # streamed, never loaded
}

def build_config(metadata: dict, tensors: dict) -> dict:
    """Qwen3.6's config fields, plus the gated residual, PLE and QSA ones (named as in transformers' config)."""
    cfg = build_qwen_config(metadata, tensors)
    arch = metadata["general.architecture"]

    def get(key, default=None):
        return metadata.get(f"{arch}.{key}", default)

    cfg.update(hc_count=get("hyper_connection.count"), hc_lowrank=get("hyper_connection.low_rank"),
               output_gate_type="sigmoid",  # not in the GGUF; Qwen3.8's config.json and llama.cpp both use sigmoid
               norm_topk_prob=True)
    ratios = [r for r in get("attention.compress_ratios", []) if r]
    cfg.update(indexer_n_heads=get("attention.indexer.head_count"), indexer_head_dim=get("attention.indexer.key_length"),
               indexer_budget=get("attention.indexer.top_k"), indexer_compress_ratio=ratios[0] if ratios else None)
    # ple.layers is 0-based in the GGUF (the HF config's ple_layer_ids are 1-based)
    cfg.update(ple_layers=list(get("ple.layers", [])), ngram_size=get("ple.ngram_size"),
               heads_per_ngram=get("ple.heads_per_ngram"), ple_conv_kernel_size=get("ple.conv_kernel"),
               ple_eos_token_id=get("ple.eos_token_id"), ple_layer_multipliers=list(get("ple.layer_multipliers", [])),
               ple_head_offsets=list(get("ple.head_offsets", [])),
               ple_head_vocab_sizes=list(get("ple.head_vocab_sizes", [])))
    if len(cfg["ple_layers"]) > 1:
        raise NotImplementedError("more than one PLE layer is not supported (the GGUF has one set of hash constants)")
    return cfg

def load_engine(gguf_path: str, metadata: dict, tensors: dict, pack_dir: str = None, cache_gb: float = None,
                preload: bool = True, gpu_backbone: bool = True, sync_timings: bool = False,
                min_expert_weight: float = 0.0, backbone: str = "int8") -> Qwen38Engine:
    """See engines/qwen36/loader.py's load_engine for the options. The non-expert weights default to int8: in bf16
    they take 9.2 GiB, too much next to the expert cache on a 16 GB Mac."""
    arch = metadata["general.architecture"]
    engine_class = ARCHITECTURES[arch]
    cfg = build_config(metadata, tensors)
    weights = load_weights(arch, engine_class.prefix, cfg, tensors, LAYER_NAMES_38, GLOBAL_NAMES_38,
                           backbone_converter(backbone))
    ple_table = StreamedEmbedding(tensors[PLE_TABLE]) if cfg["ple_layers"] else None
    return build_engine(engine_class, gguf_path, metadata, cfg, weights, pack_dir, cache_gb, preload, gpu_backbone,
                        sync_timings, min_expert_weight, ple_table=ple_table)
