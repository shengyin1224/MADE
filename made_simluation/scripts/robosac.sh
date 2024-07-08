# # no attack 
# python test_attack_det.py --detection_method robosac --log --logpath experiments/robosac/no_attack

# # 1 attacker
CUDA_VISIBLE_DEVICES=3 python test_attack_det.py --detection_method robosac --attack config/attack/single_agent/N01_E1e-01_S10.yaml --log --logpath experiments/robosac/new_N01_E1e-01_S10 --robosac_cfg config/robosac_probing.yml

CUDA_VISIBLE_DEVICES=2 python test_attack_det.py --detection_method robosac --attack config/attack/single_agent/N01_E1e-01_S10.yaml --log --logpath experiments/robosac/known_N01_E1e-01_S10 --robosac_cfg config/robosac.yml

# # 2 attackers colla
CUDA_VISIBLE_DEVICES=6 python test_attack_det.py --detection_method robosac --attack config/attack/multi_agent/collaborative/N02_E1e-01_S10.yaml --log --logpath experiments/robosac/collaborative/N02_E1e-01_S10 --robosac_cfg config/robosac_probing.yml

CUDA_VISIBLE_DEVICES=6 python test_attack_det.py --detection_method robosac --attack config/attack/multi_agent/collaborative/N02_E1e-01_S10.yaml --log --logpath experiments/robosac/collaborative/N02_E1e-01_S10 --robosac_cfg config/robosac.yml

# # 3 attackers colla
# python test_attack_det.py --detection_method robosac --attack config/attack/multi_agent/collaborative/N03_E1e-01_S10.yaml --log --logpath experiments/robosac/collaborative/N03_E1e-01_S10

# # 2 attackers non-colla
# python test_attack_det.py --detection_method robosac --attack config/attack/multi_agent/non-collaborative/N02_E1e-01_S10.yaml --log --logpath experiments/robosac/non-collaborative/N02_E1e-01_S10

# # 3 attackers non-colla
# python test_attack_det.py --detection_method robosac --attack config/attack/multi_agent/non-collaborative/N03_E1e-01_S10.yaml --log --logpath experiments/robosac/non-collaborative/N03_E1e-01_S10