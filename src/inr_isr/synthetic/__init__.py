"""Analytic structured plasma scenes; generation only, without neural models."""

from .field import StructuredScene, average_scene, scene_from_definition

__all__ = ["StructuredScene", "average_scene", "scene_from_definition"]
