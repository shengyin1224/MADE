# no attack 
python test_attack_det.py --detection_method residual_autoencoder_v2 --log --logpath experiments/residual_ae/no_attack

# 1 attacker
python test_attack_det.py --detection_method residual_autoencoder --attack config/attack/single_agent/N01_E1e-01_S10.yaml --log --logpath experiments/residual_ae/N01_E1e-01_S10

# 2 attackers colla
python test_attack_det.py --detection_method residual_autoencoder --attack config/attack/multi_agent/collaborative/N02_E1e-01_S10.yaml --log --logpath experiments/residual_ae/collaborative/N02_E1e-01_S10

# 3 attackers colla
python test_attack_det.py --detection_method residual_autoencoder --attack config/attack/multi_agent/collaborative/N03_E1e-01_S10.yaml --log --logpath experiments/residual_ae/collaborative/N03_E1e-01_S10

# 2 attackers non-colla
python test_attack_det.py --detection_method residual_autoencoder --attack config/attack/multi_agent/non-collaborative/N02_E1e-01_S10.yaml --log --logpath experiments/residual_ae/non-collaborative/N02_E1e-01_S10

# 3 attackers non-colla
python test_attack_det.py --detection_method residual_autoencoder --attack config/attack/multi_agent/non-collaborative/N03_E1e-01_S10.yaml --log --logpath experiments/residual_ae/non-collaborative/N03_E1e-01_S10