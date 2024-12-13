#!/bin/bash
echo -e "1 - By hand verification of Neural Network \n2 - SGD Neural Network \n3 - Pytorch Neural Network"
read -p 'Choice: ' choice

if [ "$choice" == 1 ]; then
    python nn-fw-bw-pass.py
fi

if [ "$choice" == 2 ]; then
    python nn_sgd.py
fi

if [ "$choice" == 3 ]; then
    python nn-pytorch.py
fi