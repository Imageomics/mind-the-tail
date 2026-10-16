"""Image dataset backed by a sample manifest (plain files or tar members)."""

from __future__ import annotations

import tarfile
import typing as tp

import polars as pl
from PIL import Image
from torch.utils.data import Dataset


class ManifestImages(Dataset):
    """Yields ``(transform(image), class_idx, example_idx)``.

    With ``skip_invalid=True`` unreadable images yield ``None`` instead of raising;
    use :func:`collate_valid` to drop them from the batch.
    """

    def __init__(self, manifest: pl.DataFrame, transform: tp.Callable, *, skip_invalid: bool = False):
        columns = [c for c in ("image_path", "class_idx", "example_idx", "tar_path", "member_name") if c in manifest.columns]
        self.rows = manifest.select(columns).to_dicts()
        self.transform = transform
        self.skip_invalid = skip_invalid
        self._tars: dict[str, tarfile.TarFile] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getstate__(self) -> dict[str, object]:
        # Open tar handles cannot be pickled into dataloader workers.
        return {**self.__dict__, "_tars": {}}

    def _open(self, row: dict[str, object]) -> Image.Image:
        tar_path = row.get("tar_path")
        if not tar_path:
            with Image.open(str(row["image_path"])) as image:
                return image.convert("RGB")
        archive = self._tars.get(str(tar_path))
        if archive is None:
            archive = self._tars[str(tar_path)] = tarfile.open(str(tar_path), "r:*")
        member = archive.extractfile(str(row["member_name"]))
        if member is None:
            raise OSError(f"Missing tar member {row['member_name']} in {tar_path}")
        with member, Image.open(member) as image:
            return image.convert("RGB")

    def __getitem__(self, index: int):
        row = self.rows[index]
        try:
            image = self.transform(self._open(row))
        except (OSError, ValueError, tarfile.TarError):
            if self.skip_invalid:
                return None
            raise
        return image, int(row["class_idx"]), int(row.get("example_idx", index))


def collate_valid(batch: list):
    """Default collate after dropping samples that failed to load."""
    from torch.utils.data import default_collate

    valid = [item for item in batch if item is not None]
    return default_collate(valid) if valid else None
