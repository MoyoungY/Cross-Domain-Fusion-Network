# Physics-aware Cross-domain Fusion Network for Computer-Generated Holography

This repository contains the official implementation of the paper ["Physics-aware Cross-domain Fusion Aids Learning-driven Computer-generated Holography"](https://github.com/computational-imaging/neural-holography) (Photonics Research 2024).

## Overview

This work proposes a novel physics-aware cross-domain fusion network (CDFN) for computer-generated holography (CGH). The network combines physical insights with deep learning to achieve high-quality holographic reconstruction.

## Features

- Physics-aware cross-domain fusion network (CDFN) implementation
- Iterative phase mapping (IPM) module
- Deep supervision (DS) strategy
- Comparison implementations of DPAC and SGD methods
- Support for RGB channels

## Installation

1. Clone this repository:
```bash
git clone https://github.com/MoyoungY/cdfn-holography.git
cd cdfn-holography
```

2. Create a conda environment:
```bash
conda create -n cdfn python=3.8
conda activate cdfn
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the CDFN model:
```bash
python main_cdfn.py --channel 1 --exp_name exp
```

Parameters:
- `--channel`: Color channel selection (0:red, 1:green, 2:blue)
- `--ipm`: Enable iterative phase mapping
- `--ds`: Enable deep supervision 
- `--exp_name`: Experiment name for logging

### Testing

To test a trained model:
```bash
python main_cdfn.py --test True --ckpt_path path/to/checkpoint
```

### Baseline Methods

Run DPAC algorithm:
```bash
python main_dpac.py
```

Run SGD algorithm:
```bash
python main_sgd.py
```

## Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{yuan2024physics,
  title={Physics-aware cross-domain fusion aids learning-driven computer-generated holography},
  author={Yuan, Ganzhangqin and Zhou, Mi and Liu, Fei and Chen, Mu Ku and Jiang, Kui and Peng, Yifan and Geng, Zihan},
  journal={Photonics Research},
  volume={12},
  number={12},
  pages={2747--2756},
  year={2024},
  publisher={Chinese Laser Press and Optica Publishing Group}
}
```

## Acknowledgments

This code are built upon the [Neural Holography](https://github.com/computational-imaging/neural-holography) repository:

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
