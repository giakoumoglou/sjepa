# S-JEPA

Official PyTorch implementation of **S-JEPA** (Stochastic Joint-Embedding Predictive Architecture), a self-supervised learning method that generalizes [I-JEPA](https://arxiv.org/abs/2301.08243) by introducing stochastic prediction in representation space. Instead of predicting a single point representation for a masked target region, S-JEPA models a **distribution over plausible target representations** conditioned on a learned stochastic latent variable `z`. This allows the model to capture **ambiguity**, produce diverse completions, and estimate predictive uncertainty, without pixel-level reconstruction. I-JEPA is trained and evaluated alongside S-JEPA throughout this repository as the deterministic baseline, using matched encoders, masking strategies, and training schedules.

> **A Stochastic Latent Variable Generalization of Joint-Embedding Predictive Architectures**  
> Nikos Giakoumoglou, Paschalis Giakoumoglou, Tania Stathaki — **WACV 2027**  
> [Paper](https://arxiv.org/pdf/XXXX.XXXXX)

---

## Overview

S-JEPA extends the original JEPA framework by replacing the deterministic treatment of the auxiliary variable `z` with a stochastic latent variable. During training, a **target-conditioned posterior** `q` infers `z` from both context and target representations, while a **context-conditioned prior** `p` learns to predict its distribution using only the visible context and is used at inference. Both networks operate entirely in representation space, without an image decoder or reconstruction objective.

<img width="1672" height="941" alt="ijepa_vs_sjepa" src="https://github.com/user-attachments/assets/920722f0-aa49-4c59-9115-4ef367b9c560" />

## ImageNet Evaluation

S-JEPA consistently learns stronger representations than I-JEPA under matched architectures and training schedules, without relying on computationally intensive data augmentations. The additional overhead is minimal, requiring only 1.01× the parameters of I-JEPA, with negligible impact on FLOPs and inference latency.

<img width="2048" height="879" alt="sjepa_evaluations" src="https://github.com/user-attachments/assets/9c7ac2e2-c6a8-49ab-aef4-da63dadce99f" />


## Visualizations

### PCA

We visualize the learned patch representations using PCA on ViT-B/16 encoder features. All three methods recover coherent object and background structures, with S-JEPA exhibiting particularly clear object boundaries and spatially consistent representations, suggesting that stochastic prediction preserves and potentially improves the semantic structure of learned features.

<img width="1536" height="993" alt="sjepa_pca" src="https://github.com/user-attachments/assets/25d4bf6a-ee31-4824-ba66-30c2984a2f5a" />

### Predictive Uncertainty

Unlike deterministic I-JEPA, which has zero predictive variance by construction, S-JEPA estimates uncertainty by sampling multiple latent variables from its learned context-conditioned prior and computing the variance of predicted target representations across samples. This produces spatially structured uncertainty maps that correlate with representation-space prediction errors.

<img width="1448" height="783" alt="sjepa_uncertainty" src="https://github.com/user-attachments/assets/6c505ee3-c4c3-4691-9cbb-a85ba002550c" />

---

## Requirements

```bash
conda create -n sjepa python=3.10
conda activate sjepa
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pyyaml numpy Pillow einops scikit-learn submitit
```

---

## Pretraining

### ViT-B/16 — 600 epochs

*Recommended: 16× A100.*

```bash
# I-JEPA
python main.py \
  --fname configs/in1k/ijepa_vitb16_in1k_ep600.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3

# S-JEPA
python main.py \
  --fname configs/in1k/sjepa_vitb16_in1k_ep600.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3
```

### ViT-L/16 — 600 epochs

*Recommended: 16× A100.*

```bash
# I-JEPA
python main.py \
  --fname configs/in1k/ijepa_vitl16_in1k_ep600.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3

# S-JEPA
python main.py \
  --fname configs/in1k/sjepa_vitl16_in1k_ep600.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3
```

### ViT-H/14 — 300 epochs

I-JEPA only, using `configs/in1k/ijepa_vith14_in1k_ep300.yaml`.

For a SLURM cluster with [submitit](https://github.com/facebookincubator/submitit):

```bash
python main_distributed.py \
  --fname configs/in1k/ijepa_vith14_in1k_ep300.yaml \
  --folder $path_to_save_submitit_logs \
  --partition $slurm_partition \
  --nodes 2 --tasks-per-node 8 \
  --time 1000
```

---

## License

See the [LICENSE](./LICENSE) file for details.

## Citation

If you find this repository useful, please consider giving it a star ⭐ and citing our paper:

```bibtex
@inproceedings{giakoumoglou2027sjepa,
  title={A Stochastic Latent Variable Generalization of Joint-Embedding Predictive Architectures},
  author={Giakoumoglou, Nikos and Giakoumoglou, Paschalis and Stathaki, Tania},
  booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2027}
}
```
