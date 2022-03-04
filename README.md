# Continuous-Time Meta-Learning with Forward Mode Differentiation

[**ICLR 2022 (Spotlight)**](https://arxiv.org/abs/2203.01443) - [**Installation**](#installation) - [**Example**](COMLN%20-%20LEO%20miniImageNet.ipynb) - [**Citation**](#citation)

This repository contains the official implementation in [JAX](https://github.com/google/jax) of **COMLN** ([Deleu et al., 2022](https://arxiv.org/abs/2203.01443)), a gradient-based meta-learning algorithm, where adaptation follows a gradient flow. It contains an implementation of the [memory-efficient algorithm](comln/utils/gradient_flow.py) to compute the meta-gradients, based on forward-mode differentiation. The implementation is based on [jax-meta](https://github.com/tristandeleu/jax-meta-learning).

## Installation
To avoid any conflict with your existing Python setup, we are suggesting to work in a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```
Follow these [instructions](https://github.com/google/jax#installation) to install the version of JAX corresponding to your versions of CUDA and CuDNN. Note that if you want to test COMLN from the [example notebook](COMLN%20-%20LEO%20miniImageNet.ipynb), you must also install [Jupyter notebook](https://jupyter.org/).
```bash
git clone https://github.com/tristandeleu/jax-comln.git
cd jax-comln
pip install -r requirements.txt
```

## Citation
If you want to cite COMLN, use the following Bibtex entry:
```
@inproceedings{deleu2022comln,
    title={{Continuous-Time Meta-Learning with Forward Mode Differentiation}},
    author={Deleu, Tristan and Kanaa, David and Feng, Leo and Kerg, Giancarlo and Bengio, Yoshua and Lajoie, Guillaume and Bacon, Pierre-Luc},
    booktitle={Tenth International Conference on Learning Representations},
    year={2022}
}
```
