# GRADMM

This repository is the official implementation of our ICML 2025 paper [Synthetic Text Generation for Training Large Language Models via Gradient Matching](https://arxiv.org/pdf/2502.17607).

## 🔗 Quick Links
- [GRADMM](#gradmm)
  - [🔗 Quick Links](#-quick-links)
  - [Install Requirements](#install-requirements)
  - [Data Generation](#data-generation)
  - [Finetuning](#finetuning)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgements)


## Install Requirements
```bash
conda create -n gradmm python=3.11
conda activate gradmm
pip install -r requirements.txt
```

## Data Generation
```bash
cd gradmm
./scripts/admm.sh
./scripts/admm_dp.sh
```

For filtering, please refer to the notebook `gradmm/Filtering.ipynb`. Adjust the settings in the `Parameters` section, then run all cells in the notebook.

## Finetuning
1. Obtain the synthetic data paths by running the `Print fine-tuning paths` section in the notebook `gradmm/Finetuning.ipynb`.

2. Insert the retrieved paths into `scripts/query_ft.sh`, then run the following commands:
```bash
cd addax
./scripts/query_ft.sh
```

3. To collect the fine-tuning results, paste the fine-tuning paths into `Collect fine-tuning results` section in the notebook `gradmm/Finetuning.ipynb` and un the corresponding cells.

## Bugs or Questions?
If you have any questions related to the code or the paper, feel free to email Dang Nguyen (nguyentuanhaidang@gmail.com). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation
Please cite our paper if you find the repo helpful in your work:

```bibtex
@article{nguyen2025synthetic,
  title={Synthetic Text Generation for Training Large Language Models via Gradient Matching},
  author={Nguyen*, Dang and Li*, Zeman and Bateni, Mohammadhossein and Mirrokni, Vahab and Razaviyayn, Meisam and Mirzasoleiman, Baharan},
  journal={International Conference on Machine Learning (ICML)},
  year={2025}
}
```

## Acknowledgements
The structure of this repository is largely based on the official implementation of [lamp](https://github.com/eth-sri/lamp) and [Addax](https://github.com/optimization-for-data-driven-science/Addax). We are grateful for their open sources.