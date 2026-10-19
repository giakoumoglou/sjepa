# S-JEPA

Official PyTorch implementation of **S-JEPA** (Stochastic Joint-Embedding Predictive Architecture), a self supervised learning method that generalizes [I-JEPA](https://arxiv.org/abs/2301.08243) by replacing its deterministic predictor with a stochastic one. Instead of predicting a single point representation for a masked target region, the model predicts a distribution over representations conditioned on a learned latent variable $z$. This lets it capture genuine ambiguity about a masked region, representing several plausible completions rather than being forced to average them into one, while keeping prediction entirely in representation space with no pixel level reconstruction. I-JEPA is trained and evaluated alongside S-JEPA throughout this repository as the baseline, using the exact same encoder, masking, and data pipeline.

> **A Stochastic Latent Variable Generalization of Joint-Embedding Predictive Architectures** ([arXiv](https://arxiv.org/abs/XXXX.XXXXX))  
> Nikos Giakoumoglou, Paschalis Giakoumoglou, Tania Stathaki — WACV 2027

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

*Recommended: 16x A100.*

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

*Recommended: 16x A100.*

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

### ViT-H/14 — 300 epochs (I-JEPA only, `configs/in1k/ijepa_vith14_in1k_ep300.yaml`)

For a cluster with a SLURM + [submitit](https://github.com/facebookincubator/submitit) setup rather than PBS:

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
See the [LICENSE](./LICENSE) file for details about the license under which this code is made available.

## Citation
If you find this repository useful in your research, please consider giving a star :star: and a citation:
```
@article{giakoumoglou2026sjepa,
  title={A Stochastic Latent Variable Generalization of Joint-Embedding Predictive Architectures},
  author={Giakoumoglou, Nikos and Giakoumoglou, Paschalis and Stathaki, Tania},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```
