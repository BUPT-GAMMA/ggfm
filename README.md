![GGFM](framework.png) is an open-source pipeline for graph foundation model based on PyTorch. We integrate SOTA graph foundation models.

The relevant documentation is available [here](https://ggfm.readthedocs.io/en/latest/).
It is under development, welcome join us!

Install
============

System requrements
------------------
GGFM works with the following operating systems:

* Linux


Python environment requirments
------------------------------

- [Python](https://www.python.org/) >= 3.8
- [PyTorch](https://pytorch.org/get-started/locally/) >= 2.1.0
- [DGL](https://github.com/dmlc/dgl) >= 2.0.0
- [PyG](https://www.pyg.org/) >= 2.4.0

**1. Python environment (Optional):** We recommend using Conda package manager

```bash
conda create -n ggfm python=3.8
source activate ggfm
```

**2. Pytorch:** Follow their [tutorial](https://pytorch.org/get-started/) to run the proper command according to
your OS and CUDA version. For example:

```bash
pip install torch torchvision torchaudio
```

**3. DGL:** Follow their [tutorial](https://www.dgl.ai/pages/start.html) to run the proper command according to
your OS and CUDA version. For example:

```bash
pip install dgl -f https://data.dgl.ai/wheels/repo.html
```

**4. PyG:** Follow their [tutorial](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) to run the proper command according to
your OS and CUDA version. For example:

```bash
pip install torch_geometric
```

**4. Install GGFM:**

* install from pypi

```bash
pip install ggfm
```

* install from source

```bash
git clone https://github.com/BUPT-GAMMA/ggfm
cd ggfm
pip install .
```
