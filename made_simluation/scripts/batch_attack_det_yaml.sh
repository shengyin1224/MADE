#!/bin/bash

# python test_attack_det.py --attack config/attack/single_agent/N01_E5e-01_S10.yaml --detection_method ours --log --logpath experiments/ours/single/N01_E5e-01_S10

# python test_attack_det.py --attack config/attack/single_agent/N01_E5e-02_S10.yaml --detection_method ours --log --logpath experiments/ours/single/N01_E5e-02_S10

# python test_attack_det.py --attack config/attack/multi_agent/collaborative/N02_E5e-01_S10.yaml --detection_method ours --log --logpath experiments/ours/collaborative/N02_E5e-01_S10

python test_attack_det.py --attack config/attack/multi_agent/collaborative/N02_E5e-02_S10.yaml --detection_method ours --log --logpath experiments/ours/collaborative/N02_E5e-02_S10

# python test_attack_det.py --attack config/attack/multi_agent/non-collaborative/N02_E5e-01_S10.yaml --detection_method ours --log --logpath experiments/ours/non-collaborative/N02_E5e-01_S10

# python test_attack_det.py --attack config/attack/multi_agent/non-collaborative/N02_E5e-02_S10.yaml --detection_method ours --log --logpath experiments/ours/non-collaborative/N02_E5e-02_S10

python test_attack_det.py --attack config/attack/single_agent/N01_E5e-01_S10.yaml --detection_method binary_classifier --log --logpath experiments/binary_classifier/single/N01_E5e-01_S10

python test_attack_det.py --attack config/attack/single_agent/N01_E5e-02_S10.yaml --detection_method binary_classifier --log --logpath experiments/binary_classifier/single/N01_E5e-02_S10

# python test_attack_det.py --attack config/attack/multi_agent/collaborative/N02_E5e-01_S10.yaml --detection_method binary_classifier --log --logpath experiments/binary_classifier/collaborative/N02_E5e-01_S10

# python test_attack_det.py --attack config/attack/multi_agent/collaborative/N02_E5e-02_S10.yaml --detection_method binary_classifier --log --logpath experiments/binary_classifier/collaborative/N02_E5e-02_S10

python test_attack_det.py --attack config/attack/multi_agent/non-collaborative/N02_E5e-01_S10.yaml --detection_method binary_classifier --log --logpath experiments/binary_classifier/non-collaborative/N02_E5e-01_S10

python test_attack_det.py --attack config/attack/multi_agent/non-collaborative/N02_E5e-02_S10.yaml --detection_method binary_classifier --log --logpath experiments/binary_classifier/non-collaborative/N02_E5e-02_S10