#!/bin/bash

python test_attack_det.py --detection_method none --attack config/attack/gt/single_agent/N01_E5e-02_S10.yaml --log --logpath experiments/attack/gt/single/N01_E5e-02_S10

python test_attack_det.py --detection_method none --attack config/attack/gt/multi_agent/non-collaborative/N02_E5e-02_S10.yaml  --log --logpath experiments/attack/gt/non-collaborative/N02_E5e-02_S10

python test_attack_det.py --detection_method none --attack config/attack/gt/multi_agent/non-collaborative/N03_E5e-02_S10.yaml  --log --logpath experiments/attack/gt/non-collaborative/N03_E5e-02_S10