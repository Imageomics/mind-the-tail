#!/usr/bin/env python3
"""Write nested top-k% WebDataset subsets of CC12M from a global ranking.

One scan over the source tar shards writes every requested ratio. Only the source
shard name and sample key are taken from each ranked ``image_path``
(``tar://<dir>/<shard>.tar::<key>.jpg``), so the source tars may have moved.

    python clip/build_subset.py --ranking OUT/ranked_examples.parquet \
        --source-dir ${DATA_ROOT}/cc12m-wds --output ${DATA_ROOT}/cc12m_subsets/ours --ratios 10 20 30 50 70
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import pyarrow.parquet as pq


def selected_masks(ranking: Path, ratios: list[int]) -> tuple[dict[str, dict[str, int]], dict[int, int], int]:
    """``{shard: {key: bitmask of ratios containing the sample}}`` plus per-ratio counts."""
    table = pq.ParquetFile(ranking)
    total = table.metadata.num_rows
    counts = {ratio: total * ratio // 100 for ratio in ratios}
    wanted: dict[str, dict[str, int]] = defaultdict(dict)
    for batch in table.iter_batches(columns=["rank_global", "image_path"], batch_size=250_000):
        for rank, image_path in zip(*batch.to_pydict().values()):
            if rank >= counts[ratios[-1]]:
                continue
            tar_part, member = image_path.rsplit("::", 1)
            shard, key = Path(tar_part.removeprefix("tar://")).stem, PurePosixPath(member).stem
            wanted[shard][key] = sum(1 << i for i, ratio in enumerate(ratios) if rank < counts[ratio])
    return wanted, counts, total


class Writers:
    """One rolling tar writer per ratio, ``samples_per_shard`` samples per output tar."""

    def __init__(self, root: Path, ratios: list[int], samples_per_shard: int):
        self.root, self.ratios, self.per_shard = root, ratios, samples_per_shard
        self.counts = {ratio: 0 for ratio in ratios}
        self.open: dict[int, tuple[int, tarfile.TarFile]] = {}
        for ratio in ratios:
            (root / f"top_{ratio}pct" / "shards").mkdir(parents=True)

    def archive(self, ratio: int) -> tarfile.TarFile:
        index = self.counts[ratio] // self.per_shard
        current = self.open.get(ratio)
        if current is None or current[0] != index:
            if current is not None:
                current[1].close()
            path = self.root / f"top_{ratio}pct" / "shards" / f"{index:08d}.tar"
            self.open[ratio] = (index, tarfile.open(path, "w"))
        return self.open[ratio][1]

    def add(self, mask: int, member: tarfile.TarInfo, payload: bytes) -> None:
        for i, ratio in enumerate(self.ratios):
            if mask >> i & 1:
                self.archive(ratio).addfile(member, io.BytesIO(payload))

    def finish_sample(self, mask: int) -> None:
        for i, ratio in enumerate(self.ratios):
            self.counts[ratio] += mask >> i & 1

    def close(self) -> None:
        for _, archive in self.open.values():
            archive.close()


def build(ranking: Path, source_dir: Path, output: Path, ratios: list[int], samples_per_shard: int) -> None:
    wanted, counts, total = selected_masks(ranking, ratios)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    writers = Writers(tmp, ratios, samples_per_shard)
    try:
        for n_done, shard in enumerate(sorted(wanted), 1):
            keys, found, current, mask = wanted[shard], set(), None, 0
            with tarfile.open(source_dir / f"{shard}.tar") as archive:
                # WebDataset groups all files of a sample (jpg, txt, json) consecutively.
                for member in archive:
                    if not member.isfile():
                        continue
                    key = PurePosixPath(member.name).stem
                    if key != current:
                        writers.finish_sample(mask)
                        current, mask = key, keys.get(key, 0)
                        if mask:
                            found.add(key)
                    if mask:
                        writers.add(mask, member, archive.extractfile(member).read())
                writers.finish_sample(mask)
            if found != set(keys):
                raise ValueError(f"{shard}: {len(set(keys) - found)} selected samples missing")
            if n_done % 100 == 0:
                print(f"{n_done}/{len(wanted)} source shards", flush=True)
        writers.close()
        if writers.counts != counts:
            raise ValueError(f"wrote {writers.counts}, expected {counts}")
        for ratio in ratios:
            meta = {"ranking": str(ranking), "ranking_count": total, "top_percent": ratio,
                    "selected_count": counts[ratio],
                    "output_shard_count": -(-counts[ratio] // samples_per_shard)}
            (tmp / f"top_{ratio}pct" / "selection_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        tmp.replace(output)
    except BaseException:
        writers.close()
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ratios", type=int, nargs="+", default=[10, 20, 30, 50, 70])
    parser.add_argument("--samples-per-shard", type=int, default=10_000)
    args = parser.parse_args()
    build(args.ranking, args.source_dir, args.output, sorted(set(args.ratios)), args.samples_per_shard)


if __name__ == "__main__":
    main()
