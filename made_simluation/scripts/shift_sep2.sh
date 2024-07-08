# python test_attack_det.py --detection_method match_cost_v2 --attack config/attack/shift/N01_E1e-1_S10_sep.yaml --log --logpath experiments/match_cost_v2/ --att_subpath shift/N01_E1e-1_S10_sep

# python test_attack_det.py --detection_method residual_autoencoder_v2 --attack config/attack/shift/N01_E1e-1_S10_sep.yaml --log --logpath experiments/residual_ae_v2/ --att_subpath shift/N01_E1e-1_S10_sep

# python test_attack_det.py --detection_method multi-test-raev2 --attack config/attack/shift/N01_E1e-1_S10_sep.yaml --log --logpath experiments/multi-test-raev2/ --att_subpath shift/N01_E1e-1_S10_sep

# python test_attack_det.py --detection_method match_cost_v2 --attack config/attack/shift/multi_agent/non-collaborative/N02_E1e-1_S10_sep.yaml --log --logpath experiments/match_cost_v2/ --att_subpath shift/non-collaborative/N02_E1e-1_S10_sep

# python test_attack_det.py --detection_method residual_autoencoder_v2 --attack config/attack/shift/multi_agent/non-collaborative/N02_E1e-1_S10_sep.yaml --log --logpath experiments/residual_ae_v2/ --att_subpath shift/non-collaborative/N02_E1e-1_S10_sep

# python test_attack_det.py --detection_method multi-test-raev2 --attack config/attack/shift/multi_agent/non-collaborative/N02_E1e-1_S10_sep.yaml --log --logpath experiments/multi-test-raev2/ --att_subpath shift/non-collaborative/N02_E1e-1_S10_sep

python test_attack_det.py --detection_method match_cost_v2 --attack config/attack/shift/multi_agent/collaborative/N02_E1e-1_S10_sep.yaml --log --logpath experiments/match_cost_v2/ --att_subpath shift/collaborative/N02_E1e-1_S10_sep

python test_attack_det.py --detection_method residual_autoencoder_v2 --attack config/attack/shift/multi_agent/collaborative/N02_E1e-1_S10_sep.yaml --log --logpath experiments/residual_ae_v2/ --att_subpath shift/collaborative/N02_E1e-1_S10_sep

python test_attack_det.py --detection_method multi-test-raev2 --attack config/attack/shift/multi_agent/collaborative/N02_E1e-1_S10_sep.yaml --log --logpath experiments/multi-test-raev2/ --att_subpath shift/collaborative/N02_E1e-1_S10_sep