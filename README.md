<div align="center">

# Example code for BUFF (WiP)


[![arXiv](https://img.shields.io/badge/arXiv-2404.18219-b31b1b.svg)](https://arxiv.org/abs/2404.18219)
[![python](https://img.shields.io/badge/-Python_3.9-blue?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch_1.8+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![xgboost](https://img.shields.io/badge/xgboost-2.0.3-brightgreen.svg?logo=git&logoColor=white)](https://xgboost.readthedocs.io/en/stable/install.html)



</div>


**Abstract**:

> Tabular data stands out as one of the most frequently encountered types in high energy physics. Unlike commonly homogeneous data such as pixelated images, simulating high-dimensional tabular data and accurately capturing their correlations are often quite challenging, even with the most advanced architectures. Based on the findings that tree-based models surpass the performance of deep learning models for tasks specific to tabular data, we adopt the very recent generative modeling class named conditional flow matching and employ different techniques to integrate the usage of Gradient Boosted Trees. The performances are evaluated for various tasks on different analysis level with several public datasets. We demonstrate the training and inference time of most high-level simulation tasks can achieve speedup by orders of magnitude. The application can be extended to low-level feature simulation and conditioned generations with competitive performance.


## How to run

```bash
# install required packages
pip install -r requirements.txt

Run a simple minimal example here
editting or adding your preprocessing code to the folder `preprocessing`

example running code in the folder `runner`
you may want to use multicore training on HTCondor `runner/htcondor`


## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{Jiang:2024bwr,
    author = "Jiang, Cheng and Qian, Sitian and Qu, Huilin",
    title = "{BUFF: Boosted Decision Tree based Ultra-Fast Flow matching}",
    eprint = "2404.18219",
    archivePrefix = "arXiv",
    primaryClass = "physics.ins-det",
    month = "4",
    year = "2024"
}
