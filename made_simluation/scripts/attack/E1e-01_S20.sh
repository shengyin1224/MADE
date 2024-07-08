#!/bin/bash

python test_attack_det.py --detection_method none --attack config/attack/gt/single_agent/N01_E1e-01_S20.yaml --log --logpath experiments/attack/gt/single/N01_E1e-01_S20

python test_attack_det.py --detection_method none --attack config/attack/gt/multi_agent/non-collaborative/N02_E1e-01_S20.yaml  --log --logpath experiments/attack/gt/non-collaborative/N02_E1e-01_S20

python test_attack_det.py --detection_method none --attack config/attack/gt/multi_agent/non-collaborative/N03_E1e-01_S20.yaml  --log --logpath experiments/attack/gt/non-collaborative/N03_E1e-01_S20