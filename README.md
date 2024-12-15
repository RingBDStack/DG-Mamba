# [AAAI 2025] DG-Mamba: Robust and Efficient Dynamic Graph Structure Learning with Selective State Space Models

This repository is the official implementation of "[DG-Mamba: Robust and Efficient Dynamic Graph Structure Learning with Selective State Space Models]()" accepted by the Main Technical Track of the 39th Annual AAAI Conference on Artificial Intelligence (AAAI-2025).

<p align="center">
<img src="framework.png" width="100%" class="center" alt="logo"/>
</p>

-----

## 0. Abstract

Dynamic graphs exhibit intertwined spatio-temporal evolutionary patterns, widely existing in the real world. Nevertheless, the structure incompleteness, noise, and redundancy result in poor robustness for Dynamic Graph Neural Networks (DGNNs). Dynamic Graph Structure Learning (DGSL) offers a promising way to optimize graph structures. However, aside from encountering unacceptable quadratic complexity, it overly relies on heuristic priors, making it hard to discover underlying predictive patterns. How to efficiently refine the dynamic structures, capture intrinsic dependencies, and learn robust representations, remains under-explored. In this work, we propose the novel DG-Mamba, a robust and efficient Dynamic Graph structure learning framework with the Selective State Space Models (Mamba). To accelerate the spatio-temporal structure learning, we propose a kernelized dynamic message-passing operator that reduces the quadratic time complexity to linear. To capture global intrinsic dynamics, we establish the dynamic graph as a self-contained system with State Space Model. By discretizing the system states with the cross-snapshot graph adjacency, we enable the long-distance dependencies capturing with the selective snapshot scan. To endow learned dynamic structures more expressive with informativeness, we propose the self-supervised Principle of Relevant Information for DGSL to regularize the most relevant yet least redundant information, enhancing global robustness. Extensive experiments demonstrate the superiority of the robustness and efficiency of our DG-Mamba compared with the state-of-the-art baselines against adversarial attacks.

## 1. Requirements

Main package requirements:

- `CUDA == 10.1`
- `Python == 3.8.12`
- `PyTorch == 1.9.1`
- `PyTorch-Geometric == 2.0.1`

To install the complete requiring packages, use the following command at the root directory of the repository:

```setup
pip install -r requirements.txt
```

## 2. Quick Start

### Training

To train DG-Mamba, run the following command in the ROOT directory :

```train
python main.py --dataset collab --load_best_config
```

Or:

```
python main.py --dataset collab --lr 0.0025 --weight_decay 1e-3 --num_layers 1 --hidden_channels 32 --num_heads 1 --rb_order 1 --rb_trans sigmoid --M 30 --K 10 --use_bn --use_residual --use_gumbel --epochs 2000 --beta1 0.1 --beta2 50.0 --gamma 0.0025 --lamda_1 0.025 --mu 1.0 --patience 500
```

### Evaluation

To evaluate DG-Mamba with trained models, run the following command in the ROOT directory:

```eval
python eval.py --dataset=<dataset_name>  --exp_type=<exp_mode>  --load_best_config
```

Please download the pre-trained model from **link** and put the trained model in the directory `./saved_model`

Example:

```
python eval.py --dataset=yelp  --exp_type=feature  --load_best_config
python eval.py --dataset=yelp_evasive_1  --exp_type=evasive  --load_best_config
```

#### Explanations for the arguments:

- `dataset_name`: name of the datasets, including "collab", "yelp", and "act".
- `load_best_config`: if training with the preset configurations. 
- `exp_type`: adversarial attacking modes, including "clean", "structure", "feature", "evasive", and "poisoning".

#### Configurations for `dataset_name` under different attacking modes

- Non-targeted adversarial attack: `dataset_name` is chosen from "collab", "yelp", and "act".
- Targeted adversarial attack (evasive): `dataset_name` is chosen from "collab_evasive_1", "collab_evasive_2", "collab_evasive_3", "collab_evasive_4", "yelp_evasive_1", "yelp_evasive_2", "yelp_evasive_3", "yelp_evasive_4", "act_evasive_1", "act_evasive_2", "act_evasive_3", "act_evasive_4".
- Targeted adversarial attack (poisoning): `dataset_name` is chosen from "collab_poisoning_1", "collab_poisoning_2", "collab_poisoning_3", "collab_poisoning_4", "yelp_poisoning_1", "yelp_poisoning_2", "yelp_poisoning_3", "yelp_poisoning_4", "act_poisoning_1", "act_poisoning_2", "act_poisoning_3", "act_poisoning_4".

## 3. Citation
If you find this repository helpful, please consider citing us (arXiv version). We welcome any discussions with [yuanhn@buaa.edu.cn](mailto:yuanhn@buaa.edu.cn).

```bibtex
@article{yuan2024dg,
  title={DG-Mamba: Robust and Efficient Dynamic Graph Structure Learning with Selective State Space Models},
  author={Yuan, Haonan and Sun, Qingyun and Wang, Zhaonan and Fu, Xingcheng and Ji, Cheng and Wang, Yongjian and Jin, Bo and Li, Jianxin},
  journal={arXiv preprint arXiv:2412.08160},
  year={2024}
}
```
