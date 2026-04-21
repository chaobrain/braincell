# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""On-disk cache for downloaded NeuroMorpho.Org neurons.

A cache is a directory containing one subfolder per cached neuron, named
by ``neuron_id``. Each subfolder holds the file(s) downloaded by
:class:`NeuroMorphoClient.download` plus a ``metadata.json`` describing
the upstream record. :class:`NeuroMorphoCache` discovers, queries, and
loads cached neurons; :class:`NeuroMorphoCacheLayout` is the
side-effect-free path builder used internally.
"""



import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from .models import NeuroMorphoCacheStatus, NeuroMorphoMeasurement
from .urls import safe_filename

if TYPE_CHECKING:
    from braincell.io.swc.types import SwcReport
    from braincell.morph._morphology import Morphology

__all__ = [
    "NeuroMorphoCache",
    "NeuroMorphoCacheLayout",
]


@dataclass(frozen=True)
class NeuroMorphoCacheLayout:
    """Pure path builder for the on-disk NeuroMorpho.Org cache layout.

    Performs no I/O. Answers "where would file *X* for neuron *N* live
    if it existed". Used by :class:`NeuroMorphoCache` and testable in
    isolation.

    Parameters
    ----------
    root : Path
        Cache root directory. Need not exist; the layout is purely
        symbolic.

    Examples
    --------

    .. code-block:: python

        >>> layout = NeuroMorphoCacheLayout(Path("/tmp/nm"))
        >>> layout.neuron_dir(10047)
        PosixPath('/tmp/nm/10047')
        >>> layout.metadata_path(10047)
        PosixPath('/tmp/nm/10047/metadata.json')
    """

    root: Path

    def neuron_dir(self, neuron_id: int) -> Path:
        """Return the per-neuron folder under :attr:`root`."""

        return self.root / str(int(neuron_id))

    def metadata_path(self, neuron_id: int) -> Path:
        """Return the path of ``metadata.json`` for *neuron_id*."""

        return self.neuron_dir(neuron_id) / "metadata.json"

    def standard_swc_path(self, neuron_id: int, neuron_name: str) -> Path:
        """Return the path of the standardized SWC file.

        Parameters
        ----------
        neuron_id : int
        neuron_name : str
            The neuron name (used to derive the file stem).
        """

        stem = safe_filename(neuron_name)
        return self.neuron_dir(neuron_id) / f"{stem}.CNG.swc"

    def original_file_path(
        self,
        neuron_id: int,
        neuron_name: str,
        suffix: str,
    ) -> Path:
        """Return the path of the original (Source-Version) file.

        Parameters
        ----------
        neuron_id : int
        neuron_name : str
        suffix : str
            File extension including the leading dot, e.g. ``".asc"``.
        """

        stem = safe_filename(neuron_name)
        return self.neuron_dir(neuron_id) / f"{stem}{suffix}"


class NeuroMorphoCache:
    """First-class cache over locally-downloaded NeuroMorpho.Org neurons.

    Owns a :class:`NeuroMorphoCacheLayout` and provides discovery and
    loading operations. Replaces the loose ``find_standard_swc`` and
    ``load_cached_metadata`` helpers from previous versions.

    Parameters
    ----------
    root : str or Path
        Directory containing one subfolder per cached neuron. The
        directory is created lazily on first write.

    Attributes
    ----------
    root : Path
    layout : NeuroMorphoCacheLayout

    Examples
    --------

    .. code-block:: python

        >>> cache = NeuroMorphoCache("~/data/neuromorpho")
        >>> for neuron_id in cache.list_neurons():
        ...     morph = cache.load(neuron_id)  # doctest: +SKIP
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        self.layout = NeuroMorphoCacheLayout(root=self.root)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_neurons(self) -> tuple[int, ...]:
        """Return the sorted ids of every cached neuron.

        Returns
        -------
        tuple of int
            Empty tuple when the cache directory does not exist.
        """

        if not self.root.exists():
            return ()
        ids: list[int] = []
        for entry in self.root.iterdir():
            if not entry.is_dir():
                continue
            try:
                ids.append(int(entry.name))
            except ValueError:
                continue
        ids.sort()
        return tuple(ids)

    def contains(self, neuron_id: int) -> bool:
        """Return ``True`` if a folder for *neuron_id* exists on disk."""

        return self.layout.neuron_dir(neuron_id).is_dir()

    def status(self, neuron_id: int) -> NeuroMorphoCacheStatus:
        """Return a :class:`NeuroMorphoCacheStatus` snapshot for *neuron_id*.

        Parameters
        ----------
        neuron_id : int

        Returns
        -------
        NeuroMorphoCacheStatus
        """

        folder = self.layout.neuron_dir(neuron_id)
        metadata_file = self.layout.metadata_path(neuron_id)
        exists = folder.exists()
        if not exists:
            return NeuroMorphoCacheStatus(
                configured=True,
                folder=folder,
                exists=False,
                metadata_exists=False,
                standard_exists=False,
                original_exists=False,
                neuron_id=int(neuron_id),
            )

        metadata_exists = metadata_file.exists()
        standard_exists = False
        original_exists = False

        if metadata_exists:
            try:
                metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            except FileNotFoundError:
                metadata = {}
            except json.JSONDecodeError:
                metadata = {}
            neuron_name = metadata.get("neuron_name")
            if isinstance(neuron_name, str):
                std = self.layout.standard_swc_path(neuron_id, neuron_name)
                standard_exists = std.exists()
                fmt = metadata.get("original_format")
                if isinstance(fmt, str):
                    suffix = Path(fmt).suffix
                    if suffix:
                        original_exists = self.layout.original_file_path(
                            neuron_id, neuron_name, suffix
                        ).exists()

        if not standard_exists:
            standard_exists = any(folder.glob("*.CNG.swc"))

        return NeuroMorphoCacheStatus(
            configured=True,
            folder=folder,
            exists=True,
            metadata_exists=metadata_exists,
            standard_exists=standard_exists,
            original_exists=original_exists,
            neuron_id=int(neuron_id),
        )

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def metadata(self, neuron_id: int) -> Mapping[str, Any]:
        """Return the parsed ``metadata.json`` for *neuron_id*.

        Parameters
        ----------
        neuron_id : int

        Returns
        -------
        Mapping[str, Any]

        Raises
        ------
        FileNotFoundError
            If the metadata file does not exist.
        """

        path = self.layout.metadata_path(neuron_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def measurement(self, neuron_id: int) -> NeuroMorphoMeasurement | None:
        """Return the typed measurement for *neuron_id*, if cached.

        Parameters
        ----------
        neuron_id : int

        Returns
        -------
        NeuroMorphoMeasurement or None
            ``None`` when no metadata file is present or it does not
            contain a ``measurement`` block.
        """

        try:
            metadata = self.metadata(neuron_id)
        except FileNotFoundError:
            return None
        payload = metadata.get("measurement")
        if not isinstance(payload, Mapping):
            return None
        if "neuron_id" not in payload:
            payload = dict(payload)
            payload.setdefault("neuron_id", int(neuron_id))
        return NeuroMorphoMeasurement.from_payload(payload)

    def standard_swc_path(self, neuron_id: int) -> Path | None:
        """Return the path of the cached standardized SWC file.

        Resolves the path by reading ``metadata.json`` for the neuron
        name when available, otherwise falls back to ``*.CNG.swc`` then
        ``*.swc`` glob inside the per-neuron folder.

        Parameters
        ----------
        neuron_id : int

        Returns
        -------
        Path or None
            ``None`` if no SWC file can be located.
        """

        folder = self.layout.neuron_dir(neuron_id)
        if not folder.exists():
            return None

        try:
            metadata = self.metadata(neuron_id)
        except FileNotFoundError:
            metadata = {}

        for item in metadata.get("download_items", []) or []:
            if not isinstance(item, Mapping) or item.get("kind") != "standard":
                continue
            raw_path = item.get("path")
            if raw_path:
                candidate = Path(raw_path)
                try:
                    candidate.resolve().relative_to(folder.resolve())
                    if candidate.exists():
                        return candidate
                except ValueError:
                    pass
                candidate = folder / Path(raw_path).name
                if candidate.exists():
                    return candidate
            filename = item.get("filename")
            if filename:
                candidate = folder / str(filename)
                if candidate.exists():
                    return candidate

        neuron_name = metadata.get("neuron_name")
        if isinstance(neuron_name, str):
            candidate = self.layout.standard_swc_path(neuron_id, neuron_name)
            if candidate.exists():
                return candidate

        cng_matches = sorted(folder.glob("*.CNG.swc"))
        if cng_matches:
            return cng_matches[0]
        swc_matches = sorted(folder.glob("*.swc"))
        return swc_matches[0] if swc_matches else None

    def original_file_path(self, neuron_id: int) -> Path | None:
        """Return the path of the cached original (Source-Version) file.

        Parameters
        ----------
        neuron_id : int

        Returns
        -------
        Path or None
            ``None`` if no original file is present in the folder.
        """

        folder = self.layout.neuron_dir(neuron_id)
        if not folder.exists():
            return None
        try:
            metadata = self.metadata(neuron_id)
        except FileNotFoundError:
            metadata = {}

        for item in metadata.get("download_items", []) or []:
            if not isinstance(item, Mapping) or item.get("kind") != "original":
                continue
            if item.get("skip"):
                continue
            raw_path = item.get("path")
            if raw_path:
                candidate = Path(raw_path)
                try:
                    candidate.resolve().relative_to(folder.resolve())
                    if candidate.exists():
                        return candidate
                except ValueError:
                    pass
                candidate = folder / Path(raw_path).name
                if candidate.exists():
                    return candidate
            filename = item.get("filename")
            if filename:
                candidate = folder / str(filename)
                if candidate.exists():
                    return candidate
        return None

    def load(
        self,
        neuron_id: int,
        *,
        mode: str | None = "neuromorpho",
        return_report: bool = False,
    ) -> "Morphology | tuple[Morphology, SwcReport]":
        """Load a cached neuron as a :class:`Morphology`.

        Parses the cached standardized SWC file with
        :meth:`Morphology.from_swc`.

        Parameters
        ----------
        neuron_id : int
        mode : str or None
            SWC import mode forwarded to :meth:`Morphology.from_swc`.
            Defaults to ``"neuromorpho"``.
        return_report : bool
            If ``True``, also return the :class:`SwcReport`.

        Returns
        -------
        Morphology or (Morphology, SwcReport)

        Raises
        ------
        FileNotFoundError
            If no standardized SWC file is cached for *neuron_id*.
        """

        from braincell.morph._morphology import Morphology

        path = self.standard_swc_path(neuron_id)
        if path is None:
            raise FileNotFoundError(
                f"No cached standardized SWC found for neuron_id={neuron_id} "
                f"under {self.root}"
            )
        return Morphology.from_swc(path, mode=mode, return_report=return_report)

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def remove(self, neuron_id: int) -> bool:
        """Delete the on-disk folder for *neuron_id*.

        Parameters
        ----------
        neuron_id : int

        Returns
        -------
        bool
            ``True`` if a folder was removed, ``False`` if there was
            nothing to remove.
        """

        folder = self.layout.neuron_dir(neuron_id)
        if not folder.exists():
            return False
        shutil.rmtree(folder)
        return True

    def clear(self) -> int:
        """Delete every per-neuron folder under :attr:`root`.

        Does not delete files at the root level that are not neuron
        folders.

        Returns
        -------
        int
            Number of folders that were removed.
        """

        if not self.root.exists():
            return 0
        removed = 0
        for entry in self.root.iterdir():
            if not entry.is_dir():
                continue
            try:
                int(entry.name)
            except ValueError:
                continue
            try:
                shutil.rmtree(entry)
                removed += 1
            except OSError as exc:
                raise OSError(f"Failed to remove cache folder {entry}: {exc}") from exc
        return removed
