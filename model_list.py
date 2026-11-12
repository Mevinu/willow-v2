"""models.json: the models willow knows by name, so commands can take a name instead of a path and the web page
can switch between them. The file sits next to main.py:

    {
      "default": "Qwen3.8-Flash-Next",
      "models": [
        {"name": "Qwen3.8-Flash-Next", "path": "/path/to/model-00001-of-00004.gguf", "cache_gb": 4},
        {"name": "Qwen3.6-35B-A3B", "path": "/path/to/model.gguf"}
      ]
    }

Each model needs a name and a path (a .gguf file; for a split GGUF, any one of its files). It can also set pack_dir,
cache_gb and backbone, which apply when the matching command-line flag isn't given. default is the model `web`
starts with when none is named (default: the first one)."""
import json
import os

MODELS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.json")
SETTINGS = ("pack_dir", "cache_gb", "backbone")  # optional per-model settings

def read_models(path: str = MODELS_FILE):
    """-> (models [{"name", "path", and any SETTINGS}], the default model's name). ([], None) when there's no file."""
    if not os.path.exists(path):
        return [], None
    with open(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path} is not valid JSON: {e}")
    models, names = [], set()
    for i, entry in enumerate(data.get("models", [])):
        if not isinstance(entry, dict) or not entry.get("name") or not entry.get("path"):
            raise SystemExit(f"{path}: model {i + 1} needs a \"name\" and a \"path\"")
        if entry["name"] in names:
            raise SystemExit(f"{path}: two models are named {entry['name']!r}")
        unknown = set(entry) - {"name", "path", *SETTINGS}
        if unknown:
            raise SystemExit(f"{path}: model {entry['name']!r} has unknown settings: {', '.join(sorted(unknown))} "
                             f"(allowed: {', '.join(SETTINGS)})")
        names.add(entry["name"])
        models.append(dict(entry, path=os.path.expanduser(entry["path"])))
    default = data.get("default")
    if default is not None and default not in names:
        raise SystemExit(f"{path}: the default model {default!r} isn't one of its models")
    return models, default or (models[0]["name"] if models else None)

def resolve(model: str, models: list, default: str) -> dict:
    """A model name from models.json (any case) or a path -> its entry. With no model, the default one."""
    if model is None:
        if default is None:
            raise SystemExit(f"No model given, and {MODELS_FILE} lists none: pass a .gguf file, or add models there")
        model = default
    for entry in models:
        if entry["name"].lower() == model.lower():
            return entry
    for entry in models:  # a path that models.json lists keeps its name and settings
        if os.path.abspath(entry["path"]) == os.path.abspath(os.path.expanduser(model)):
            return entry
    if not model.endswith(".gguf"):
        names = ", ".join(entry["name"] for entry in models) or "none"
        raise SystemExit(f"{model} is neither a .gguf file nor a model in models.json (its models: {names})")
    return {"name": os.path.basename(model).removesuffix(".gguf"), "path": os.path.expanduser(model)}
