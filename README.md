# VLGPO: Variational Latent Generative Protein Optimization

**ICML 2025**

This repository contains the inference code for [A Variational Perspective on Generative Protein Fitness Optimization](https://arxiv.org/abs/2501.19200) accepted at [ICML 2025](https://icml.cc/virtual/2025/poster/44530).

[Lea Bogensperger](https://scholar.google.com/citations?user=4cNGQ0sAAAAJ&hl=en)<sup>1</sup>,
[Dominik Narnhofer](https://scholar.google.com/citations?user=tFx8AhkAAAAJ&hl=en)<sup>2</sup>, 
[Ahmed Allam](https://scholar.google.com/citations?user=xcuCdJUAAAAJ&hl=en&oi=sra)<sup>1</sup>, 
[Konrad Schindler](https://scholar.google.com/citations?user=FZuNgqIAAAAJ&hl=en&oi=ao)<sup>2</sup>, 
[Michael Krauthammer](https://scholar.google.com/citations?user=Cgq2_M0AAAAJ&hl=en&oi=ao)<sup>1</sup>

<sup>1</sup>University of Zurich, 
<sup>2</sup>ETH Zurich

<img src="assets/vlgpo.png" alt="" width="400"/>

---

## 🛠️ Setup

Checkpoints for the predictors $g_\phi$ and $g_{\tilde{\phi}}$ (classifier guidance) and the in-silico oracle $g_\psi$ (evaluation) are taken from [GGS: Gibbs sampling with Graph-based Smoothing](https://github.com/kirjner/GGS). 

### 📦 Repository

The file `requirements.txt` contains a list of the Python packages required to run this project.

### 🚀 Run code

```bash
# 1. Create and activate the environment
conda create -n vlgpo-env python=3.11
conda activate vlgpo-env
pip install -r requirements.txt

# 2. Run the sampler
python src/vlgpo/sample.py
```

## 🎓 Citation

```bibtex
@article{bogensperger2025variational,
  title={A Variational Perspective on Generative Protein Fitness Optimization},
  author={Bogensperger, Lea and Narnhofer, Dominik and Allam, Ahmed and Schindler, Konrad and Krauthammer, Michael},
  journal={arXiv preprint arXiv:2501.19200},
  year={2025}
}