# SCVulBERT

Model Training Part Smart Contract Vul for Detection based on Pretrain Bert (torch backen)

## Paper

Paper "A High-Performance Smart Contract Vulnerability Detection Scheme Based on BERT" has published in ICPADS 2023.

Author: Shengqiang Zeng, Ruhuang Chen, Hongwei Zhang, and Jinsong Wang.

Online Link https://ieee-cybermatics.org/2023/icpads/icpads-2023-accepted.htm .

Paper Manuscript already upload as [SCVulBert.6.17.2023](https://github.com/PoilZero/SCVulBert/blob/main/SCVulBert.6.17.2023.pdf).

## Requirements

### Required Packages

* python3
* torch1（https://pytorch.org/get-started/locally/）
* transformers

### Required Dataset

This repository contains the smart contract dataset of source code and processed code fragment. As to the source code, we crawled the source code of the smart contract from the [Ethereum](https://etherscan.io/) by using the crawler tool. Besides, we have collected some data from some other websites. At the same time, we also designed and wrote some smart contract codes with reentrancy vulnerabilities.

**Note:** crawler tool is available [here](https://github.com/Messi-Q/Crawler).

### Dataset

Original Ethereum smart contracts: [Etherscan_contract](https://drive.google.com/open?id=1h9aFFSsL7mK4NmVJd4So7IJlFj9u0HRv)

The train data after normalization in `dataset`.

## OverView

### Model

In our experiment, we modify the last layer to a two-classification task using a pre-train Bert-base model.

Tokenization and Hyperparameters are the same for the baseline (bert-base-cased), which hyperparameters from `E50_com.py` are used.

Implementation is very basic without many optimizations, so it is easier to debug and play around with the code.

* hyperparameters please edit E50_com.py

### Trainning

You can directly run with the E65 file, and you can also specify the dataset with first parameter, this will replace the E50 hyperparameters setting.

In train.sh will run

* python E65_train.py
* python E65_train.py dataset/reentrancy_273.txt
* sh train.sh

## References

* Rechecker for Normalized dataset: https://github.com/Messi-Q/ReChecker
  * This Project already stops maintained, this is a branch that I keep maintaining.
  * https://github.com/PoilZero/ReChecker

## Citation

```
@inproceedings{zeng2023solgpt,
  title={SolGPT: A GPT-Based Static Vulnerability Detection Model for Enhancing Smart Contract Security},
  author={Zeng, Shengqiang and Zhang, Hongwei and Wang, Jinsong and Shi, Kai},
  booktitle={International Conference on Algorithms and Architectures for Parallel Processing},
  pages={42--62},
  year={2023},
  organization={Springer}
}
@inproceedings{zeng2023high,
  title={A High-Performance Smart Contract Vulnerability Detection Scheme Based on BERT},
  author={Zeng, Shengqiang and Chen, Ruhuang and Zhang, Hongwei and Wang, Jinsong},
  booktitle={2023 IEEE 29th International Conference on Parallel and Distributed Systems (ICPADS)},
  pages={653--658},
  year={2023},
  organization={IEEE}
}
```
