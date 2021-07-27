# HEMP
HEMP: High order Entropy Minimizationfor neural network comPression
[![DOI](https://zenodo.org/badge/doi/10.1016/j.neucom.2021.07.022.svg)](http://dx.doi.org/10.1016/j.neucom.2021.07.022)
[![arXiv](https://img.shields.io/badge/arXiv-2102.03773-b31b1b.svg)](https://arxiv.org/abs/2107.05298)

Please cite this work as

```latex
@article{TARTAGLIONE2021,
title = {HEMP: High-order Entropy Minimization for neural network comPression},
journal = {Neurocomputing},
year = {2021},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2021.07.022},
url = {https://www.sciencedirect.com/science/article/pii/S0925231221010663},
author = {Enzo Tartaglione and Stephane Lathuiliere and Attilio Fiandrotti and Marco Cagnazzo and Marco Grangetto}
}

```

## Requirements
* PyTorch >= 1.8.1
* CUDA >= 11.1
* scipy >= 1.5.4
* numpy >= 1.20.2
* torchvision >= 0.9.1
* py7zr >= 0.16.0
* matplotlib >= 3.2.2
* tqdm >= 4.56.0

## Running code
```
python3 main.py \
-model [architecture] \
-dataset [training dataset] \
-device [cuda:id or cpu] \
-batch_size [batch size for training] \
-test-batch-size [batch size for test] \
-epochs [wall epochs]\
-lr [learning rate] \
-lamb_H [weight on HEMP] \
-lamb_RMSE [weight on RMSE term] \
-entropy_order [entropy order to be evaluated] \
-N [number of bins]
```

