"""Resolve a published GorillaWatch checkpoint into a config + local weights file."""

import json
from pathlib import Path

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"


def resolve_hf_checkpoint(model_ref, revision=None, token=None):
    """Return (config_dict, path_to_model.safetensors) for a published checkpoint.

    ``model_ref`` is either a HuggingFace repo id (e.g.
    "gorilla-watch/GorillaWatch-DINOv2-Large") or a local directory containing
    ``config.json`` and ``model.safetensors``. Hub files are fetched through the
    usual huggingface_hub cache, so repeated evaluations do not re-download.
    """
    local = Path(model_ref)
    if local.is_dir():
        config_path = local / CONFIG_NAME
        weights_path = local / WEIGHTS_NAME
        missing = [p.name for p in (config_path, weights_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{local} is missing {missing}")
    else:
        from huggingface_hub import hf_hub_download

        download_kwargs = dict(
            repo_id=str(model_ref),
            repo_type="model",
            revision=revision,
            token=token,
        )
        config_path = Path(hf_hub_download(filename=CONFIG_NAME, **download_kwargs))
        weights_path = Path(hf_hub_download(filename=WEIGHTS_NAME, **download_kwargs))

    return json.loads(config_path.read_text()), str(weights_path)
