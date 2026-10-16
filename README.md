# Mind the Tail: Data Selection Through Sparse Concepts

This repository contains the code for **Mind the Tail**: concept extraction with a sparse autoencoder (SAE) on DINOv3 features, subset selection, and downstream training and evaluation. The Mind the Tail website is hosted from the gh-pages branch of this repository.

[Paper](https://arxiv.org/pdf/2610.21007) | [Project Page](https://imageomics.github.io/mind-the-tail/) | [Models & Data](https://huggingface.co/imageomics/mind-the-tail-data)
---

Mind the Tail selects training subsets by tracking how much support each visual concept receives, shifting the budget toward under-supported concepts that are shared by neighboring samples. It improves over strong data selection baselines on fine-grained classification, spurious-correlation benchmarks, and CLIP pretraining on CC12M.


## Table of Contents

1. [Models & Data](#models-data)
2. [Commands](#commands)
3. [Paper, Website, and Data](#paper)
4. [Citation](#citation)

<h2 id="models-data">Models & Data</h2>

We release the trained SAE, the selection rankings, and the downstream checkpoints, so the pipeline can be started at any step. All of them are in our [Hugging Face repository](https://huggingface.co/imageomics/mind-the-tail-data).

- **SAE**: TopK-32 SAE on DINOv3-B layer-11 patch tokens, trained on ImageNet-1K.
- **Selection rankings**: `ranked_examples.parquet` for NABirds, Stanford Cars, Waterbirds, Spawrious, and CC12M.
- **Models**: ResNet-50 checkpoints for the classification datasets and CLIP ViT-B/16 checkpoints for CC12M.

<h2 id="commands">Commands</h2>

### Setup

Clone this repository, then install the requirements:

```bash
git clone https://github.com/Imageomics/mind-the-tail.git
cd mind-the-tail
pip install -e .            # SAE, selection, ResNet-50
pip install -e ".[clip]"    # CLIP experiments
```

**DINOv3 weights.** DINOv3 is a gated model. Request access on the [DINOv3-B model page](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) (approved by Meta), then log in so the weights can be downloaded:

```bash
huggingface-cli login
```

**Data.** Download the datasets from their original sources: [ImageNet-1K](https://www.image-net.org/), [NABirds](https://dl.allaboutbirds.org/nabirds), Stanford Cars, [Waterbirds](https://github.com/kohpangwei/group_DRO) (`waterbird_complete95_forest2water2`), [Spawrious](https://github.com/aengusl/spawrious) (one-to-one hard), [CC12M](https://github.com/google-research-datasets/conceptual-12m) as WebDataset shards, and [Flickr30K](https://shannon.cs.illinois.edu/DenotationGraph/) (Karpathy split). Set `DATA_ROOT` to the dataset folder and `OUTPUT_ROOT` to where outputs should go; the configs expect:

```
${DATA_ROOT}/
├── imagenet/{train,val}/<wnid>/*.JPEG
├── nabirds/{images/, images.txt, classes.txt, image_class_labels.txt, train_test_split.txt}
├── stanford_cars/{train,test}/<class name>/*.jpg
├── waterbirds/{metadata.csv, <species dirs>/*.jpg}
├── spawrious/spawrious224/{0,1}/<background>/<breed>/*.png
├── cc12m-wds/cc12m-train-*.tar
└── flickr30k/{images/, flickr_annotations_30k.csv}
```

### Pipeline

The full pipeline runs in five steps. Steps 1 and 2 train the SAE once on ImageNet-1K; steps 3 to 5 run per dataset. The examples use NABirds; replace `nabirds` with `stanford_cars`, `waterbirds`, `spawrious`, or `imagenet1k` for the other datasets.

**Step 1. Extract DINOv3 features.** Extract DINOv3-B layer-11 patch tokens of ImageNet-1K and shuffle them across shards for SAE training.

```bash
python -m sae.extract --config configs/sae/extract_imagenet1k_train.yaml
python -m sae.extract --config configs/sae/extract_imagenet1k_val.yaml
python -m sae.remap   --config configs/sae/remap_imagenet1k_train.yaml
```

**Step 2. Train the SAE.** Train a TopK SAE on 100M patch tokens and evaluate it on the held-out tokens.

```bash
python -m sae.train    --config configs/sae/train_topk32.yaml
python -m sae.evaluate --config configs/sae/evaluate.yaml
```

**Step 3. SAE inference.** Encode every image of the candidate pool into one sparse concept vector. To use the released SAE instead of steps 1 and 2, add `--set checkpoint=<path to sae.pt>`.

```bash
python -m sae.infer images --config configs/sae/infer_images_nabirds.yaml
```

**Step 4. Data selection.** Rank the candidate pool with our objective. A subset of ratio `r` is the top of this ranking.

```bash
python -m selection.select --config configs/selection/nabirds.yaml
```

**Step 5. Downstream training and evaluation.** Train a ResNet-50 on the selected subset and evaluate it on the test split (with worst-group accuracy for Waterbirds and Spawrious). Repeat with `seed=1` and `seed=2`, then average the three runs.

```bash
python -m downstream.train    --config configs/downstream/nabirds.yaml \
    --set ratio=0.1 seed=0 output_dir=${OUTPUT_ROOT}/downstream/nabirds_10pct/seed0
python -m downstream.evaluate --config configs/downstream/eval_nabirds.yaml \
    --set checkpoint=${OUTPUT_ROOT}/downstream/nabirds_10pct/seed0/checkpoints/best.pt \
          output_dir=${OUTPUT_ROOT}/downstream/nabirds_10pct/seed0/eval
python -m downstream.aggregate --runs ${OUTPUT_ROOT}/downstream/nabirds_10pct/seed*/eval/summary.json \
    --output ${OUTPUT_ROOT}/downstream/nabirds_10pct/summary.json
```

**CC12M.** Steps 3 to 5 for CLIP pretraining: infer concept vectors of the WebDataset shards, rank the 11M-image pool in one pass, build the subsets, train CLIP ViT-B/16, and evaluate ImageNet-1K zero-shot and linear-probe accuracy and Flickr30K retrieval.

```bash
python -m sae.infer images --config configs/sae/infer_images_cc12m.yaml
python -m selection.large_scale --activations ${OUTPUT_ROOT}/image_activations/cc12m \
    --output ${OUTPUT_ROOT}/selection/cc12m --beta 0.05 --k 3
python clip/build_subset.py --ranking ${OUTPUT_ROOT}/selection/cc12m/ranked_examples.parquet \
    --source-dir ${DATA_ROOT}/cc12m-wds --output ${OUTPUT_ROOT}/cc12m_subsets
SUBSET_DIR=${OUTPUT_ROOT}/cc12m_subsets/top_10pct RATIO=10 bash clip/train_clip.sh
cd clip
CKPT=${OUTPUT_ROOT}/clip/cc12m-ours-top10pct-vitb16/checkpoints/epoch_15.pt
python eval_zeroshot.py     --checkpoint $CKPT --val-dir ${DATA_ROOT}/imagenet/val
python eval_linear_probe.py --checkpoint $CKPT --imagenet ${DATA_ROOT}/imagenet
python eval_flickr30k.py    --checkpoint $CKPT --root ${DATA_ROOT}/flickr30k
```

The same steps are wrapped in `pipeline/01_extract.sh` to `pipeline/05_train_downstream.sh`, and in Slurm job templates under `slurm/<dataset>/` (fill in the partition and account first):

```bash
DATASET=nabirds RATIO=0.1 bash pipeline/05_train_downstream.sh
DATA_ROOT=... OUTPUT_ROOT=... RATIO=0.1 sbatch --array=0-2 slurm/nabirds/train.sh
DATA_ROOT=... OUTPUT_ROOT=... RATIO=0.1 sbatch slurm/nabirds/eval.sh
```

<h2 id="paper">Paper, Website, and Data</h2>

The paper is available on [arXiv](https://arxiv.org/abs/2610.21007), and we also have a [project website](https://imageomics.github.io/mind-the-tail/). The SAE, selection rankings, and trained models are published on [Hugging Face](https://huggingface.co/imageomics/mind-the-tail-data). All datasets are available from their original sources linked above.

## Citation

Please cite our paper if you use our code or results.

```
@article{zhang2026mindthetail,
  title   = {Mind the Tail: Data Selection Through Sparse Concepts},
  author  = {Zhang, Ziheng and Hulkund, Neha and Lee, Colin and Mai, Zheda and
             Shen, Hui and Ding, Ying and Berger-Wolf, Tanya and Beery, Sara and
             Chao, Wei-Lun and Gu, Jianyang},
  journal = {arXiv preprint arXiv:2610.21007},
  year    = {2026}
}
```

Also consider citing [DINOv3](https://github.com/facebookresearch/dinov3), [saev](https://github.com/Imageomics/saev), and [OpenCLIP](https://github.com/mlfoundations/open_clip).

## License

Mind the Tail is released under the MIT License. Some elements of the code are copyright by others, including the [saev](https://github.com/Imageomics/saev) and [OpenCLIP](https://github.com/mlfoundations/open_clip) authors (see [`LICENSE`](LICENSE)).
