# vrapi

<img src="example.png?raw=true" width="450"/>

This repository contains the code for [Visualizing Representations of Adversarially Perturbed
Inputs](https://arxiv.org/abs/2105.14116), our paper that proposes a metric for evaluating 2D
dimensionality reduction algorithms for visualizing the representations of adversarially perturbed
inputs.

Reported running times are approximate, intended to give a general idea of how long each step will
take. Estimates are based on times encountered while developing on Ubuntu 20.10 with hardware that
includes an AMD Ryzen 7 2700X CPU, 64GB of memory, and an NVIDIA TITAN RTX GPU with 24GB of memory.

### Requirements

The code was developed using Python 3.9 on Ubuntu 20.10. Other systems and Python versions may work,
but have not been tested.

Python library dependencies are specified in [requirements.txt](requirements.txt). Versions are
pinned for reproducibility.

### Installation

- Optionally create and activate a virtual environment.

```shell
python3 -m venv env
source env/bin/activate
```

- Install Python dependencies, specified in `requirements.txt`.
  * 2 minutes

```shell
pip3 install -r requirements.txt
```

### Running the Code

By default, output is saved to the `./workspace` directory, which is created automatically.

- Train a ResNet classification model.
  * 50 minutes

```shell
python3 src/train_net.py
```

- Evaluate the model, extracting representations from the corresponding data.
  * Less than 1 minute

```shell
python3 src/eval_net.py
```

- Adversarially perturb test images, evaluating and extracting representations from the
  corresponding data.
  * 7 hours

```shell
python3 src/attack.py
```

- Project representations to two dimensions. Specify `TF_DETERMINISTIC_OPS` so that ParametricUMAP
  is deterministic. Also disable GPU with `CUDA_VISIBLE_DEVICES`, since at least one required
  deterministic op, `unsorted segment reduction op`, is not available for GPU.
  * 10 hours

```shell
TF_DETERMINISTIC_OPS=1 CUDA_VISIBLE_DEVICES= python3 src/projection.py
```

- Score the projections.
  * 1.5 hours

```shell
python3 src/score.py
```

- Generate visualizations.
  * 35 minutes
  * The `--approximate` option can be used to reduce running time (to less than 1 minute), but the
    consequence is that some classes take visual precedence over others in the visualizations.

```shell
python3 src/visualize.py
```

### Citation

```
@misc{steinberg2021visualizing,
      title={Visualizing Representations of Adversarially Perturbed Inputs},
      author={Daniel Steinberg and Paul Munro},
      year={2021},
      eprint={2105.14116},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
