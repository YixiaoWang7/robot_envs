"""
Feature schema for the robot dataset pipeline.

``FeatureDef``  — one feature to load (key, shape, source, temporal window).
``FeatureConfig`` — the full list of features the dataloader should provide.
"""

from __future__ import annotations

from typing import Any, Iterator, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, model_validator


# ---------------------------------------------------------------------------
# TimestampSchema (used by alignment_precompute)
# ---------------------------------------------------------------------------

class TimestampSchema(BaseModel):
    """HDF5 timestamp path conventions for the small-files dataset format."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    actions_timestamps_path: str = "modalities/actions/timestamps"
    robot_state_timestamps_path: str = "modalities/robot_state/timestamps"
    env_state_timestamps_path: str = "modalities/env_state/timestamps"
    image_timestamps_template: str = "modalities/images/{cam}/timestamps"
    legacy_actions_timestamps_path: str = "timestamps"


# ---------------------------------------------------------------------------
# FeatureDef
# ---------------------------------------------------------------------------

class FeatureDef(BaseModel):
    """One logical feature visible to the model and data pipeline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    shape: Tuple[int, ...]
    source_type: Literal["hdf5", "video"] = "hdf5"
    paths: Tuple[str, ...] = ()
    camera_names: Tuple[str, ...] = ()
    history: int = 0
    future: int = 0

    @model_validator(mode="after")
    def _check(self) -> "FeatureDef":
        if not self.key:
            raise ValueError("Feature key must not be empty.")
        if not self.shape or any(d <= 0 for d in self.shape):
            raise ValueError(f"Feature '{self.key}': shape must be non-empty with positive dims, got {self.shape}.")
        if self.history < 0 or self.future < 0:
            raise ValueError(f"Feature '{self.key}': history and future must be >= 0.")
        return self

    @property
    def is_hdf5(self) -> bool:
        return self.source_type == "hdf5"

    @property
    def is_video(self) -> bool:
        return self.source_type == "video"

    @property
    def current_idx(self) -> int:
        """0-based index of the current timestep within the output window."""
        return self.history

    @property
    def window_size(self) -> int:
        """Total number of timesteps: history + current + future."""
        return self.history + 1 + self.future


# ---------------------------------------------------------------------------
# FeatureConfig
# ---------------------------------------------------------------------------

class FeatureConfig(BaseModel):
    """List of features the dataloader should provide.

    Supports key-based lookup: ``cfg["actions"]`` returns the FeatureDef
    whose key is ``"actions"``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    features: Tuple[FeatureDef, ...]

    @model_validator(mode="after")
    def _check(self) -> "FeatureConfig":
        keys = [fd.key for fd in self.features]
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        if dupes:
            raise ValueError(f"Duplicate feature keys: {dupes}")
        return self

    # -- dict-like access by key ---------------------------------------------

    def __getitem__(self, key: str) -> FeatureDef:
        for fd in self.features:
            if fd.key == key:
                return fd
        raise KeyError(f"No feature with key {key!r}")

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and any(fd.key == key for fd in self.features)

    def __iter__(self) -> Iterator[FeatureDef]:
        return iter(self.features)

    def __len__(self) -> int:
        return len(self.features)

    # -- helpers used by dataset / processor ---------------------------------

    @property
    def max_future(self) -> int:
        """Largest future extent across all features (steps after current)."""
        return max(fd.future for fd in self.features)

    def validate_hdf5(self, demo_group: Any) -> None:
        """Check that HDF5 feature shapes match declarations."""
        for fd in self.features:
            if not fd.is_hdf5:
                continue
            actual = _infer_hdf5_shape(fd, demo_group)
            if actual != tuple(fd.shape):
                raise ValueError(
                    f"Shape mismatch for '{fd.key}': declared {fd.shape}, HDF5 has {actual}."
                )

    def validate_video(self, cam: str, frame: Any) -> None:
        """Check that a decoded video frame matches the declared shape for video features.

        Args:
            cam: Camera name (e.g. ``"agentview"``).
            frame: A single decoded frame as a numpy array with shape ``(H, W, C)``.
        """
        for fd in self.features:
            if not fd.is_video:
                continue
            if cam not in fd.camera_names:
                continue
            # fd.shape is (C, H, W); frame is (H, W, C)
            declared_c, declared_h, declared_w = fd.shape
            actual_h, actual_w, actual_c = frame.shape[:3]
            if (actual_c, actual_h, actual_w) != (declared_c, declared_h, declared_w):
                raise ValueError(
                    f"Video shape mismatch for feature '{fd.key}' cam='{cam}': "
                    f"declared (C,H,W)=({declared_c},{declared_h},{declared_w}), "
                    f"but decoded frame has (H,W,C)=({actual_h},{actual_w},{actual_c}). "
                    f"Update 'shape' in the feature config to match the actual video resolution."
                )


def _infer_hdf5_shape(fd: FeatureDef, demo_group: Any) -> Tuple[int, ...]:
    parts = []
    for path in fd.paths:
        if path not in demo_group:
            raise ValueError(f"HDF5 path '{path}' (feature '{fd.key}') not found in '{demo_group.name}'.")
        parts.append(demo_group[path])
    if len(parts) == 1:
        return tuple(parts[0].shape[1:])
    concat_dim = sum(int(p.shape[1]) for p in parts)
    return (concat_dim, *tuple(parts[0].shape[2:]))
