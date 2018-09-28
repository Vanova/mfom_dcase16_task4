#!/bin/bash

source activate ai # activate conda environment
python -u run_dcase.py -m dev -p ./params/dcase.yaml > crnn.log