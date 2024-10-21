# LPZero: Language Model Zero-cost Proxy Search from Zero

This repository contains the implementation of LPZero, a framework for automatically designing zero-cost proxies for language models, achieving superior ranking consistency and performance.

## Overview

LPZero leverages genetic programming to optimize the design of zero-cost proxies, modeled as symbolic expressions. It includes a Rule-based Pruning Strategy (RPS) to enhance search efficiency by eliminating unpromising proxies early in the process.

## Features

- Automated zero-cost proxy design
- Genetic programming for proxy optimization
- Rule-based Pruning Strategy (RPS) for efficient search
- Evaluation on models like FlexiBERT, GPT-2, and LLaMA-7B

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/pprp/LPZero.git
cd LPZero
pip install -r requirements.txt
```

## Usage

Run the main script to start the search process:

```bash
python main.py
```

## Experiments

The framework has been tested on:

- **FlexiBERT**
- **GPT-2**
- **LLaMA-7B**

For detailed experimental results, refer to the `experiments` folder.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any suggestions or bug reports.

## License

This project is licensed under the MIT License.
