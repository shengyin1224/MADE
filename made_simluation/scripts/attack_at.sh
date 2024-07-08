# # no attack 
# python test_attack_det.py --detection_method none --log --logpath experiments/AT_E1e-01_S1/no_attack --resume /DB/data/yanghengzhao-1/adversarial/DiscoNet/AT_eps0.1_step1_bs4_pred/disco/epoch_100.pth

# # 1 attacker
# python test_attack_det.py --detection_method none --attack config/attack/single_agent/N01_E1e-01_S10.yaml --log --logpath experiments/AT_E1e-01_S1/N01_E1e-01_S10 --resume /DB/data/yanghengzhao-1/adversarial/DiscoNet/AT_eps0.1_step1_bs4_pred/disco/epoch_100.pth

# # 2 attackers colla
# python test_attack_det.py --detection_method none --attack config/attack/multi_agent/collaborative/N02_E1e-01_S10.yaml --log --logpath experiments/AT_E1e-01_S1/collaborative/N02_E1e-01_S10 --resume /DB/data/yanghengzhao-1/adversarial/DiscoNet/AT_eps0.1_step1_bs4_pred/disco/epoch_100.pth

# # 3 attackers colla
# python test_attack_det.py --detection_method none --attack config/attack/multi_agent/collaborative/N03_E1e-01_S10.yaml --log --logpath experiments/AT_E1e-01_S1/collaborative/N03_E1e-01_S10 --resume /DB/data/yanghengzhao-1/adversarial/DiscoNet/AT_eps0.1_step1_bs4_pred/disco/epoch_100.pth

# # 2 attackers non-colla
# python test_attack_det.py --detection_method none --attack config/attack/multi_agent/non-collaborative/N02_E1e-01_S10.yaml --log --logpath experiments/AT_E1e-01_S1/non-collaborative/N02_E1e-01_S10 --resume /DB/data/yanghengzhao-1/adversarial/DiscoNet/AT_eps0.1_step1_bs4_pred/disco/epoch_100.pth

# # 3 attackers non-colla
# python test_attack_det.py --detection_method none --attack config/attack/multi_agent/non-collaborative/N03_E1e-01_S10.yaml --log --logpath experiments/AT_E1e-01_S1/non-collaborative/N03_E1e-01_S10 --resume /DB/data/yanghengzhao-1/adversarial/DiscoNet/AT_eps0.1_step1_bs4_pred/disco/epoch_100.pth

###########
# gt attack
###########

# 1 attacker
python test_attack_det.py --detection_method none --attack config/attack/gt/single_agent/N01_E1e-01_S10.yaml --log --logpath experiments/AT_E1e-01_S1/ --resume /DB/data/yanghengzhao-1/adversarial/DiscoNet/AT_eps0.1_step1_bs4_pred/disco/epoch_100.pth  --att_subpath gt/N01_E1e-01_S10

# 2 attackers non-colla
python test_attack_det.py --detection_method none --attack config/attack/gt/multi_agent/non-collaborative/N02_E1e-01_S10.yaml --log --logpath experiments/AT_E1e-01_S1/ --resume /DB/data/yanghengzhao-1/adversarial/DiscoNet/AT_eps0.1_step1_bs4_pred/disco/epoch_100.pth  --att_subpath gt/non-collaborative/N02_E1e-01_S10

# 3 attackers non-colla
python test_attack_det.py --detection_method none --attack config/attack/gt/multi_agent/non-collaborative/N03_E1e-01_S10.yaml --log --logpath experiments/AT_E1e-01_S1/ --resume /DB/data/yanghengzhao-1/adversarial/DiscoNet/AT_eps0.1_step1_bs4_pred/disco/epoch_100.pth --att_subpath gt/non-collaborative/N03_E1e-01_S10

# 2 attackers colla
python test_attack_det.py --detection_method none --attack config/attack/gt/multi_agent/collaborative/N02_E1e-01_S10.yaml --log --logpath experiments/AT_E1e-01_S1/ --resume /DB/data/yanghengzhao-1/adversarial/DiscoNet/AT_eps0.1_step1_bs4_pred/disco/epoch_100.pth --att_subpath gt/collaborative/N02_E1e-01_S10

# 3 attackers colla
python test_attack_det.py --detection_method none --attack config/attack/gt/multi_agent/collaborative/N03_E1e-01_S10.yaml --log --logpath experiments/AT_E1e-01_S1/ --resume /DB/data/yanghengzhao-1/adversarial/DiscoNet/AT_eps0.1_step1_bs4_pred/disco/epoch_100.pth --att_subpath gt/collaborative/N03_E1e-01_S10