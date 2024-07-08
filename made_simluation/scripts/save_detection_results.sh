# python test_attack_det.py --detection_method residual_autoencoder --attack config/attack/single_agent/keep_sparse_pos/N01_E5e-01_S20_sparse_pos.yaml --save_path experiments/residual_ae/normal attack.attack_target=none

# python test_attack_det.py --detection_method residual_autoencoder --attack config/attack/single_agent/keep_sparse_pos/N01_E5e-01_S20_sparse_pos.yaml --save_path experiments/residual_ae/N01_E5e-01_S20_sparse_pos/

python test_attack_det.py --detection_method residual_autoencoder --attack config/attack/single_agent/N01_E1e-01_S10.yaml --save_path experiments/residual_ae/N01_E1e-01_S10

# python test_attack_det.py --detection_method raw_autoencoder --attack config/attack/single_agent/keep_sparse_pos/N01_E5e-01_S20_sparse_pos.yaml --save_path experiments/raw_ae/normal attack.attack_target=none

# python test_attack_det.py --detection_method raw_autoencoder --attack config/attack/single_agent/keep_sparse_pos/N01_E5e-01_S20_sparse_pos.yaml --save_path experiments/raw_ae/N01_E5e-01_S20_sparse_pos/

python test_attack_det.py --detection_method raw_autoencoder --attack config/attack/single_agent/N01_E1e-01_S10.yaml --save_path experiments/raw_ae/N01_E1e-01_S10
