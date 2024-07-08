# # no attack 
# python test_attack_det.py --detection_method raw_autoencoder --log --logpath experiments/raw_ae/no_attack

# # 1 attacker
# python test_attack_det.py --detection_method raw_autoencoder --attack config/attack/single_agent/N01_E1e-01_S10.yaml --log --logpath experiments/raw_ae/N01_E1e-01_S10

# # 2 attackers colla
# python test_attack_det.py --detection_method raw_autoencoder --attack config/attack/multi_agent/collaborative/N02_E1e-01_S10.yaml --log --logpath experiments/raw_ae/collaborative/N02_E1e-01_S10

# # 3 attackers colla
# python test_attack_det.py --detection_method raw_autoencoder --attack config/attack/multi_agent/collaborative/N03_E1e-01_S10.yaml --log --logpath experiments/raw_ae/collaborative/N03_E1e-01_S10

# # 2 attackers non-colla
# python test_attack_det.py --detection_method raw_autoencoder --attack config/attack/multi_agent/non-collaborative/N02_E1e-01_S10.yaml --log --logpath experiments/raw_ae/non-collaborative/N02_E1e-01_S10

# # 3 attackers non-colla
# python test_attack_det.py --detection_method raw_autoencoder --attack config/attack/multi_agent/non-collaborative/N03_E1e-01_S10.yaml --log --logpath experiments/raw_ae/non-collaborative/N03_E1e-01_S10

############
# gt attack
############

# 1 attacker
python test_attack_det.py --detection_method raw_autoencoder --attack config/attack/gt/single_agent/N01_E1e-01_S10.yaml --log --logpath experiments/raw_ae/ --att_subpath gt/N01_E1e-01_S10

# # 2 attackers colla
# python test_attack_det.py --detection_method raw_autoencoder --attack config/attack/gt/multi_agent/collaborative/N02_E1e-01_S10.yaml --log --logpath experiments/raw_ae/ --att_subpath gt/collaborative/N02_E1e-01_S10

# # 3 attackers colla
# python test_attack_det.py --detection_method raw_autoencoder --attack config/attack/gt/multi_agent/collaborative/N03_E1e-01_S10.yaml --log --logpath experiments/raw_ae/ --att_subpath gt/collaborative/N03_E1e-01_S10

# 2 attackers non-colla
python test_attack_det.py --detection_method raw_autoencoder --attack config/attack/gt/multi_agent/non-collaborative/N02_E1e-01_S10.yaml --log --logpath experiments/raw_ae/ --att_subpath gt/non-collaborative/N02_E1e-01_S10

# 3 attackers non-colla
python test_attack_det.py --detection_method raw_autoencoder --attack config/attack/gt/multi_agent/non-collaborative/N03_E1e-01_S10.yaml --log --logpath experiments/raw_ae/ --att_subpath gt/non-collaborative/N03_E1e-01_S10