![Python 3.10](https://img.shields.io/badge/python-3.10-green.svg)
![Pytorch 1.12.1](https://img.shields.io/badge/pytorch-1.12.1-green.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)

# LOAN
<!---
"**Location-aware Adaptive Normalization: A Deep Learning Approach for Wildfire Danger Forecasting**" by [Mohamad Hakam Shams Eddin](https://hakamshams.github.io/), [Ribana Roscher](http://rs.ipb.uni-bonn.de/people/prof-dr-ing-ribana-roscher/) and [Juergen Gall](http://pages.iai.uni-bonn.de/gall_juergen/).
-->

["**Location-aware Adaptive Normalization: A Deep Learning Approach for Wildfire Danger Forecasting**"](https://arxiv.org/abs/2212.08208) by [Mohamad Hakam Shams Eddin](https://hakamshams.github.io/), [Ribana Roscher](http://rs.ipb.uni-bonn.de/people/prof-dr-ing-ribana-roscher/) and [Juergen Gall](http://pages.iai.uni-bonn.de/gall_juergen/). Published in [IEEE Transactions on Geoscience and Remote Sensing](https://doi.org/10.1109/TGRS.2023.3285401)

### [IEEE TGRS](https://doi.org/10.1109/TGRS.2023.3285401) | [Arxiv](https://arxiv.org/abs/2212.08208)

![Example Mapping](images/figure1.jpg "Mapped ground truth from Mesh to LiDAR")

## Setup

For conda, you can install dependencies using yml file:
```
  conda env create -f environment.yml
```
or using requirements.txt:
```
  conda create --name LOAN --file requirements.txt
```
For pip:
```
  pip install -r requirements.txt
```

## Code

The code has been tested under Pytorch 1.12.1 and Python 3.10.6 on Ubuntu 20.04.5 LTS with NVIDIA GeForce RTX 3090 GPU.

The dataloader for FireCube dataset:
```
  FireCube_dataloader.py
```
For training:
```
  train.py
```
For testing:
```
  test.py
```
<br />


![Example d](images/figure2.png "Mapped ground truth from Mesh to LiDAR")

![Example d](images/figure3.jpg "Mapped ground truth from Mesh to LiDAR")

## Dataset

To train on FireCube dataset, You can download the training/testing samples from https://zenodo.org/record/6528394 (~250GB).

Compress the zip file of the datasets.tar.gz and copy the file [mean_std_train.json](data/mean_std_train.json) into the directory datasets/datasets_grl/npy/spatiotemporal

To train on another dataset, you need to create a new dataloader file like [FireCube_dataloader.py](FireCube_dataloader.py)

## Checkpoints

Pretrained models can be downloaded from [pretrained_models](pretrained_models)


### Citation
If you find our work useful in your research, please cite:

```
@ARTICLE{LOAN,
  author={Shams Eddin, Mohamad Hakam and Roscher, Ribana and Gall, Juergen},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Location-Aware Adaptive Normalization: A Deep Learning Approach for Wildfire Danger Forecasting}, 
  year={2023},
  volume={61},
  number={},
  pages={1-18},
  doi={10.1109/TGRS.2023.3285401}}

  
@article{LOAN,
  title={Location-aware Adaptive Denormalization: A Deep Learning Approach For Wildfire Danger Forecasting},
  author={Mohamad Hakam Shams Eddin and Ribana Roscher and Juergen Gall},
  journal={ArXiv},
  year={2022},
  volume={abs/2212.08208}}

```

### Acknowledgments

This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) within the Collaborative Research Centre SFB 1502/1–2022 - [DETECT](https://sfb1502.de/) - [D05](https://sfb1502.de/projects/cluster-d/d05).

### License
The code is released under MIT License. See the [LICENSE](LICENSE) file for details.
