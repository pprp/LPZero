# LPZero: Language Model Zero-cost Proxy Search from Zero

LPZero is a framework for automatically designing zero-cost proxies for language models, achieving superior ranking consistency and performance.

## Overview

LPZero leverages genetic programming to optimize the design of zero-cost proxies, modeled as symbolic expressions. It includes a Rule-based Pruning Strategy (RPS) to enhance search efficiency by eliminating unpromising proxies early in the process.

## Key Features

- Automated zero-cost proxy design for language models
- Genetic programming algorithm for proxy optimization
- Rule-based Pruning Strategy (RPS) for efficient search space exploration
- Evaluation on state-of-the-art models like FlexiBERT, GPT-2, and LLaMA-7B
- Comprehensive set of metrics for model evaluation

## Installation

1. Clone the repository:

```bash
git clone https://github.com/pprp/LPZero.git
cd LPZero
pip install -r requirements.txt
```

## Usage

How to rank? 

```bash 
METHOD=lpzero
CUDA_VISIBLE_DEVICES=2 python lpzero/runner/eval_rank_gpt2.py \
    --method $METHOD \
    --exp_name ./saved_logs/random_GPT2_wt103 --plot --get_cost \
     > ./logs/rank_corr_${METHOD}_aftersearch.log 2>&1 &
```

How to train? 

```bash 
bash scripts/run_train.sh
```

## Experiments

The framework has been tested on:

- **FlexiBERT**
- **GPT-2**
- **LLaMA-7B**

For detailed experimental results, refer to the `exps` folder.


## License

This project is licensed under the MIT License.


## Citation

If you find this work useful in your research, please consider citing:

```
@inproceedings{Dong2024LPZeroLM,
  title={LPZero: Language Model Zero-cost Proxy Search from Zero},
  author={Peijie Dong and Lujun Li and Xiang Liu and Zhenheng Tang and Xuebo Liu and Qiang Wang and Xiaowen Chu},
  year={2024},
  url={https://arxiv.org/abs/2410.04808}
}
```

## Thanks
We appreciate the contribution of https://github.com/aaronserianni/training-free-nas and https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning