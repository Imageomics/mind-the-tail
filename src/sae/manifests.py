"""Build sample manifests (one row per image) for the datasets used in the paper.

Every manifest has the columns ``example_idx, image_path, class_idx, class_name``;
Waterbirds and Spawrious add ``group_idx`` for worst-group evaluation, and
WebDataset images add ``tar_path, member_name``.
"""

from __future__ import annotations

import pathlib
import tarfile

import polars as pl

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WATERBIRDS_CLASSES = ["landbird", "waterbird"]
SPAWRIOUS_CLASSES = ["bulldog", "corgi", "dachshund", "labrador"]
SPAWRIOUS_BACKGROUNDS = ["desert", "jungle", "dirt", "mountain", "snow", "beach"]


def _is_image(name: str) -> bool:
    return pathlib.PurePath(name).suffix.lower() in IMAGE_EXTENSIONS


def _manifest(records: list[dict[str, object]], classes: list[str]) -> pl.DataFrame:
    if not records:
        raise ValueError("Dataset produced no images.")
    frame = pl.DataFrame(records).with_row_index("example_idx")
    return frame.with_columns(
        pl.col("example_idx").cast(pl.Int64),
        pl.col("class_idx").cast(pl.Int64),
        pl.col("class_idx").replace_strict(dict(enumerate(classes)), return_dtype=pl.Utf8).alias("class_name"),
    )


def imagefolder(root: pathlib.Path) -> pl.DataFrame:
    """``root/<class>/**/<image>``; classes sorted by directory name."""
    classes = sorted(path.name for path in root.iterdir() if path.is_dir())
    records = [
        {"image_path": str(path), "class_idx": class_idx}
        for class_idx, name in enumerate(classes)
        for path in sorted((root / name).rglob("*"))
        if path.is_file() and _is_image(path.name)
    ]
    return _manifest(records, classes)


def nabirds(root: pathlib.Path, split: str) -> pl.DataFrame:
    def read(name: str, columns: list[str]) -> pl.DataFrame:
        return pl.read_csv(root / name, separator=" ", has_header=False, new_columns=columns)

    frame = (
        read("images.txt", ["img_id", "filepath"])
        .join(read("image_class_labels.txt", ["img_id", "class_id"]), on="img_id")
        .join(read("train_test_split.txt", ["img_id", "is_train"]), on="img_id")
    )
    names = {}
    for line in (root / "classes.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            class_id, name = line.split(" ", 1)
            names[int(class_id)] = name
    # Only the leaf classes that carry images; ordered by NABirds class id.
    used_ids = sorted(frame["class_id"].unique().to_list())
    id_to_idx = {class_id: idx for idx, class_id in enumerate(used_ids)}
    frame = frame.filter(pl.col("is_train") == int(split == "train")).sort("filepath")
    records = [
        {"image_path": str(root / "images" / row["filepath"]), "class_idx": id_to_idx[row["class_id"]]}
        for row in frame.iter_rows(named=True)
    ]
    return _manifest(records, [names[class_id] for class_id in used_ids])


def waterbirds(root: pathlib.Path, split: str) -> pl.DataFrame:
    split_id = {"train": 0, "val": 1, "test": 2}[split]
    metadata = pl.read_csv(root / "metadata.csv").filter(pl.col("split") == split_id).sort("img_filename")
    records = [
        {
            "image_path": str(root / row["img_filename"]),
            "class_idx": int(row["y"]),
            "group_idx": 2 * int(row["y"]) + int(row["place"]),
        }
        for row in metadata.iter_rows(named=True)
    ]
    return _manifest(records, WATERBIRDS_CLASSES)


def spawrious(root: pathlib.Path, split: str) -> pl.DataFrame:
    """Spawrious one-to-one hard benchmark (``root/spawrious224``)."""
    root = root / "spawrious224"
    total = 3168
    if split == "train":
        spurious = {"bulldog": "jungle", "dachshund": "mountain", "labrador": "snow", "corgi": "desert"}
        plan = []
        for env, count in enumerate([int(0.97 * total), int(0.87 * total)]):
            plan += [(env, spurious[c], c, count) for c in ("bulldog", "dachshund", "labrador", "corgi")]
            plan += [(env, "beach", c, total - count) for c in ("bulldog", "dachshund", "labrador", "corgi")]
    else:
        test_bg = {"bulldog": "mountain", "dachshund": "snow", "labrador": "desert", "corgi": "jungle"}
        plan = [(env, test_bg[c], c, None) for env in (0, 1) for c in ("bulldog", "dachshund", "labrador", "corgi")]

    records = []
    for env, background, name, limit in plan:
        folder = root / str(env) / background / name
        paths = sorted(path for path in folder.iterdir() if path.is_file() and _is_image(path.name))
        if limit is not None:
            if len(paths) < limit:
                raise ValueError(f"{folder} has {len(paths)} images, need {limit}.")
            paths = paths[:limit]
        class_idx = SPAWRIOUS_CLASSES.index(name)
        group_idx = class_idx * len(SPAWRIOUS_BACKGROUNDS) + SPAWRIOUS_BACKGROUNDS.index(background)
        records += [{"image_path": str(p), "class_idx": class_idx, "group_idx": group_idx} for p in paths]
    return _manifest(records, SPAWRIOUS_CLASSES)


def cc12m_wds(root: pathlib.Path, shard_pattern: str = "cc12m-train-*.tar") -> pl.DataFrame:
    """Unlabelled CC12M WebDataset shards; each image member becomes one row."""
    records = []
    for tar_path in sorted(root.glob(shard_pattern)):
        with tarfile.open(tar_path, "r:*") as archive:
            records += [
                {
                    "image_path": f"tar://{tar_path}::{member.name}",
                    "tar_path": str(tar_path),
                    "member_name": member.name,
                    "class_idx": 0,
                }
                for member in archive
                if member.isfile() and _is_image(member.name)
            ]
    return _manifest(records, ["cc12m"])


def build_manifest(name: str, root: pathlib.Path, split: str | None = None) -> pl.DataFrame:
    """Dispatch on dataset name. ``imagefolder`` reads ``root/split`` (or ``root``)."""
    root = pathlib.Path(root).expanduser()
    if name == "imagefolder":
        return imagefolder(root / split if split else root)
    if name == "nabirds":
        return nabirds(root, split or "train")
    if name == "waterbirds":
        return waterbirds(root, split or "train")
    if name == "spawrious":
        return spawrious(root, split or "train")
    if name == "cc12m_wds":
        return cc12m_wds(root, split or "cc12m-train-*.tar")
    raise ValueError(f"Unknown dataset {name!r}.")


def class_names(manifest: pl.DataFrame) -> list[str]:
    mapping = manifest.select("class_idx", "class_name").unique().sort("class_idx")
    return mapping["class_name"].to_list()
