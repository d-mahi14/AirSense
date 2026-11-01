"""Persist XGBoost trees through native JSON instead of pickle memory snapshots.

Only load trusted project artifacts. Companion hashes bind the portable files to
the unchanged original pipeline; missing or stale bundles fail explicitly.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import joblib


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_portable(pipeline, path):
    """Export fitted preprocessing/wrapper plus native trees, without refitting."""
    path = Path(path)
    classifier = pipeline.named_steps.get("classifier")
    if not type(classifier).__module__.startswith("xgboost"):
        return
    tree_path = path.with_suffix(".trees.json")
    wrapper_path = path.with_suffix(".portable.joblib")
    manifest_path = path.with_suffix(".portable.json")
    classifier.save_model(tree_path)
    portable = deepcopy(pipeline)
    # All sklearn wrapper settings are retained; no XGBoost C memory snapshot
    # reaches joblib. load_model restores the native booster after unpickling.
    portable.named_steps["classifier"].__dict__.pop("_Booster")
    joblib.dump(portable, wrapper_path)
    manifest = {"format": "airsense-xgboost-json-v1", "source_sha256": _sha(path),
                "files": {p.name: _sha(p) for p in (tree_path, wrapper_path)}}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_pipeline(path):
    path = Path(path)
    manifest_path = path.with_suffix(".portable.json")
    if not manifest_path.exists():
        return joblib.load(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tree_path = path.with_suffix(".trees.json")
    wrapper_path = path.with_suffix(".portable.joblib")
    expected_files = {tree_path.name, wrapper_path.name}
    if (manifest.get("format") != "airsense-xgboost-json-v1"
            or manifest.get("source_sha256") != _sha(path)
            or set(manifest.get("files", {})) != expected_files):
        raise ValueError("Portable model bundle does not match the original pipeline")
    for component in (tree_path, wrapper_path):
        if _sha(component) != manifest["files"][component.name]:
            raise ValueError(f"Portable model checksum mismatch: {component.name}")
    pipeline = joblib.load(wrapper_path)
    classifier = pipeline.named_steps["classifier"]
    wrapper_state = classifier.__dict__.copy()
    classifier.load_model(tree_path)
    booster = classifier._Booster
    # Native loading fills attributes from booster config. Keep the exact
    # original sklearn parameter values (e.g. base_score=None) for provenance.
    classifier.__dict__.clear()
    classifier.__dict__.update(wrapper_state)
    classifier._Booster = booster
    return pipeline


def save_pipeline(pipeline, path):
    joblib.dump(pipeline, path)
    export_portable(pipeline, path)
