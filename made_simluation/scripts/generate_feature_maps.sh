#!/bin/bash
python generate_feat_maps.py \
        --split train \
        --attack config/attack/single_agent/N01_E1e-01_S10.yaml

python generate_feat_maps.py \
        --split test \
        --attack config/attack/single_agent/N01_E1e-01_S10.yaml