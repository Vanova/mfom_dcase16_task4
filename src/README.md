Multi-label MFoM framework for DCASE 2016: Task 4
-------------------------------------------------

Project architecture
====================

![Architecture of the base framework](../docs/figures/architecture_base_framework.png)


Structure of packages
=====================

```
├── README.md
├── base
│   ├── data_loader.py
│   ├── feature.py
│   ├── model.py
│   ├── pipeline.py
│   ├── trainer.py
├── data_loader
│   ├── dcase.py
│   └── dcase.pyc
├── features
│   ├── speech.py
│   └── speech.pyc
├── model
│   ├── cnn_dcase.py
│   ├── crnn_dcase.py
│   ├── mfom.py
│   ├── objectives.py
│   └── test_mfom.py
├── pipeline
│   └── dcase.py
|
├── trainer
│   ├── dcase.py
│   ├── dcase_hyper.py
│   └── mnist_hyper.py
└── utils
    ├── config.py
    ├── dirs.py
    ├── io.py
    ├── io_mlf.py
    └── metrics.py
```