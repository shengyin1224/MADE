# no attack 
# python test_attack_det.py --detection_method none --log --logpath experiments/no_defense/no_attack

# # 1 attacker
# python test_attack_det.py --detection_method none --attack config/attack/single_agent/N01_E1e-01_S10.yaml --log --logpath experiments/no_defense/N01_E1e-01_S10

# # 2 attackers colla
# python test_attack_det.py --detection_method none --attack config/attack/multi_agent/collaborative/N02_E1e-01_S10.yaml --log --logpath experiments/no_defense/collaborative/N02_E1e-01_S10

# # 3 attackers colla
# python test_attack_det.py --detection_method none --attack config/attack/multi_agent/collaborative/N03_E1e-01_S10.yaml --log --logpath experiments/no_defense/collaborative/N03_E1e-01_S10


# # 2 attackers non-colla
# python test_attack_det.py --detection_method none --attack config/attack/multi_agent/non-collaborative/N02_E1e-01_S10.yaml --log --logpath experiments/no_defense/non-collaborative/N02_E1e-01_S10

# # 3 attackers non-colla
# python test_attack_det.py --detection_method none --attack config/attack/multi_agent/non-collaborative/N03_E1e-01_S10.yaml --log --logpath experiments/no_defense/non-collaborative/N03_E1e-01_S10

#####################
# For attack curve
#####################
# # 1 attacker
# python test_attack_det.py --detection_method none --attack config/attack/single_agent/N01_E1e-01_S10.yaml attack.pgd.eps=0.05 --log --logpath experiments/no_defense/N01_E5e-02_S10

# # 2 attackers non-colla
# python test_attack_det.py --detection_method none --attack config/attack/multi_agent/non-collaborative/N02_E1e-01_S10.yaml attack.pgd.eps=0.05 --log --logpath experiments/no_defense/non-collaborative/N02_E5e-02_S10

# # 3 attackers non-colla
# python test_attack_det.py --detection_method none --attack config/attack/multi_agent/non-collaborative/N03_E1e-01_S10.yaml attack.pgd.eps=0.05 --log --logpath experiments/no_defense/non-collaborative/N03_E5e-02_S10

# # 1 attacker
# python test_attack_det.py --detection_method none --attack config/attack/single_agent/N01_E1e-01_S10.yaml attack.pgd.eps=0.2 --log --logpath experiments/no_defense/N01_E2e-01_S10

# # 2 attackers non-colla
# python test_attack_det.py --detection_method none --attack config/attack/multi_agent/non-collaborative/N02_E1e-01_S10.yaml attack.pgd.eps=0.2 --log --logpath experiments/no_defense/non-collaborative/N02_E2e-01_S10

# # 3 attackers non-colla
# python test_attack_det.py --detection_method none --attack config/attack/multi_agent/non-collaborative/N03_E1e-01_S10.yaml attack.pgd.eps=0.2 --log --logpath experiments/no_defense/non-collaborative/N03_E2e-01_S10


#############
# gt attack
#############
# 1 attacker
python test_attack_det.py --detection_method none --attack config/attack/gt/single_agent/N01_E1e-01_S10.yaml --log --logpath experiments/no_defense/gt/N01_E1e-01_S10

# 2 attackers colla
python test_attack_det.py --detection_method none --attack config/attack/gt/multi_agent/collaborative/N02_E1e-01_S10.yaml --log --logpath experiments/no_defense/gt/collaborative/N02_E1e-01_S10

# 3 attackers colla
# python test_attack_det.py --detection_method none --attack config/attack/gt/multi_agent/collaborative/N03_E1e-01_S10.yaml --log --logpath experiments/no_defense/gt/collaborative/N03_E1e-01_S10


# 2 attackers non-colla
python test_attack_det.py --detection_method none --attack config/attack/gt/multi_agent/non-collaborative/N02_E1e-01_S10.yaml --log --logpath experiments/no_defense/gt/non-collaborative/N02_E1e-01_S10

# 3 attackers non-colla
# python test_attack_det.py --detection_method none --attack config/attack/gt/multi_agent/non-collaborative/N03_E1e-01_S10.yaml --log --logpath experiments/no_defense/gt/non-collaborative/N03_E1e-01_S10