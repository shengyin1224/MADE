#!/bin/bash

# single
# python test_attack_det.py --detection_method match_cost_rm1 --attack config/attack/shift/N01_E1e-1_S10_sep.yaml --log --logpath experiments/match_cost_rm1 --att_subpath shift/N01_E1e-1_S10_sep

# non-collaborative 2 attacker
python test_attack_det.py --detection_method match_cost_rm1 --attack config/attack/shift/multi_agent/non-collaborative/N02_E1e-1_S10_sep.yaml --log --logpath experiments/match_cost_rm1 --att_subpath shift/non-collaborative/N02_E1e-1_S10_sep

# non-collaborative 3 attacker
python test_attack_det.py --detection_method match_cost_rm1 --attack config/attack/shift/multi_agent/non-collaborative/N03_E1e-1_S10_sep.yaml --log --logpath experiments/match_cost_rm1 --att_subpath shift/non-collaborative/N03_E1e-1_S10_sep

# collaborative 2 attacker
python test_attack_det.py --detection_method match_cost_rm1 --attack config/attack/shift/multi_agent/collaborative/N02_E1e-1_S10_sep.yaml --log --logpath experiments/match_cost_rm1 --att_subpath shift/collaborative/N02_E1e-1_S10_sep

# collaborative 3 attacker
python test_attack_det.py --detection_method match_cost_rm1 --attack config/attack/shift/multi_agent/collaborative/N03_E1e-1_S10_sep.yaml --log --logpath experiments/match_cost_rm1 --att_subpath shift/collaborative/N03_E1e-1_S10_sep