# # no attack 
# python test_attack_det.py --detection_method multi-test --log --logpath experiments/multi-test/no_attack

# # # 1 attacker
# python test_attack_det.py --detection_method multi-test --attack config/attack/single_agent/N01_E1e-01_S10.yaml --log --logpath experiments/multi-test-v2/N01_E1e-01_S10

# # 2 attackers colla
# python test_attack_det.py --detection_method multi-test --attack config/attack/multi_agent/collaborative/N02_E1e-01_S10.yaml --log --logpath experiments/multi-test/collaborative/N02_E1e-01_S10

# # 3 attackers colla
# python test_attack_det.py --detection_method multi-test --attack config/attack/multi_agent/collaborative/N03_E1e-01_S10.yaml --log --logpath experiments/multi-test/collaborative/N03_E1e-01_S10

# # 2 attackers non-colla
# python test_attack_det.py --detection_method multi-test --attack config/attack/multi_agent/non-collaborative/N02_E1e-01_S10.yaml --log --logpath experiments/multi-test/non-collaborative/N02_E1e-01_S10

# # 3 attackers non-colla
# python test_attack_det.py --detection_method multi-test --attack config/attack/multi_agent/non-collaborative/N03_E1e-01_S10.yaml --log --logpath experiments/multi-test/non-collaborative/N03_E1e-01_S10

############
# gt attack
############
# 1 attacker
# python test_attack_det.py --detection_method multi-test-raev2 --attack config/attack/gt/single_agent/N01_E1e-01_S10.yaml --log --logpath experiments/multi-test-raev2/ --att_subpath gt/N01_E1e-01_S10

# 2 attackers colla
# python test_attack_det.py --detection_method multi-test-raev2 --attack config/attack/gt/multi_agent/collaborative/N02_E1e-01_S10.yaml --log --logpath experiments/multi-test-raev2/ --att_subpath gt/collaborative/N02_E1e-01_S10

# 3 attackers colla
python test_attack_det.py --detection_method multi-test-raev2 --attack config/attack/gt/multi_agent/collaborative/N03_E1e-01_S10.yaml --log --logpath experiments/multi-test-raev2/ --att_subpath gt/collaborative/N03_E1e-01_S10

# 2 attackers non-colla
# python test_attack_det.py --detection_method multi-test-raev2 --attack config/attack/gt/multi_agent/non-collaborative/N02_E1e-01_S10.yaml --log --logpath experiments/multi-test-raev2/ --att_subpath gt/non-collaborative/N02_E1e-01_S10

# 3 attackers non-colla
python test_attack_det.py --detection_method multi-test-raev2 --attack config/attack/gt/multi_agent/non-collaborative/N03_E1e-01_S10.yaml --log --logpath experiments/multi-test-raev2/ --att_subpath gt/non-collaborative/N03_E1e-01_S10