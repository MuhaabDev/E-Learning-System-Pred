"""Thin wrapper so `utils.load_models` matches the folder layout in the brief."""
from src.model_loader import ModelArtifacts, load_all

__all__ = ["ModelArtifacts", "load_all"]
