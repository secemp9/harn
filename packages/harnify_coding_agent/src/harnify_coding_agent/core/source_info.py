"""Source metadata helpers for coding-agent resources."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypedDict

SourceScope = Literal["user", "project", "temporary"]
SourceOrigin = Literal["package", "top-level"]


class PathMetadata(TypedDict, total=False):
    source: str
    scope: SourceScope
    origin: SourceOrigin
    baseDir: str | None


@dataclass(slots=True)
class SourceInfo:
    path: str
    source: str
    scope: SourceScope
    origin: SourceOrigin
    baseDir: str | None = None


def create_source_info(path: str, metadata: PathMetadata | Mapping[str, object]) -> SourceInfo:
    return SourceInfo(
        path=path,
        source=str(metadata["source"]),
        scope=metadata.get("scope", "temporary"),  # type: ignore[arg-type]
        origin=metadata.get("origin", "top-level"),  # type: ignore[arg-type]
        baseDir=str(metadata["baseDir"]) if metadata.get("baseDir") is not None else None,
    )


def create_synthetic_source_info(
    path: str,
    options: PathMetadata | Mapping[str, object],
) -> SourceInfo:
    return SourceInfo(
        path=path,
        source=str(options["source"]),
        scope=options.get("scope", "temporary"),  # type: ignore[arg-type]
        origin=options.get("origin", "top-level"),  # type: ignore[arg-type]
        baseDir=str(options["baseDir"]) if options.get("baseDir") is not None else None,
    )


createSourceInfo = create_source_info
createSyntheticSourceInfo = create_synthetic_source_info

__all__ = [
    "PathMetadata",
    "SourceInfo",
    "SourceOrigin",
    "SourceScope",
    "createSourceInfo",
    "createSyntheticSourceInfo",
    "create_source_info",
    "create_synthetic_source_info",
]
