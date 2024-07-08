#!/bin/bash

# python test_attack_det.py --detection_method none --attack config/attack/gt/single_agent/N01_E1e-01_S10.yaml --log --logpath experiments/attack/gt/single/N01_E1e-01_S10

# python test_attack_det.py --detection_method none --attack config/attack/gt/multi_agent/collaborative/N02_E1e-01_S10.yaml  --log --logpath experiments/attack/gt/collaborative/N02_E1e-01_S10

python test_attack_det.py --detection_method none --attack config/attack/gt/multi_agent/collaborative/N03_E1e-01_S10.yaml  --log --logpath experiments/attack/gt/collaborative/N03_E1e-01_S10

python test_attack_det.py --detection_method none --attack config/attack/gt/multi_agent/non-collaborative/N02_E1e-01_S10.yaml  --log --logpath experiments/attack/gt/non-collaborative/N02_E1e-01_S10

python test_attack_det.py --detection_method none --attack config/attack/gt/multi_agent/non-collaborative/N03_E1e-01_S10.yaml  --log --logpath experiments/attack/gt/non-collaborative/N03_E1e-01_S10

# # AT
# python test_attack_det.py --resume /DB/data/yanghengzhao-1/adversarial/DiscoNet/AT/disco/epoch_100.pth --detection_method none --attack config/attack/single_agent/N01_E5e-02_S10.yaml --log --logpath experiments/AT_E1e-01_S1/single/N01_E5e-02_S10

# python test_attack_det.py --resume /DB/data/yanghengzhao-1/adversarial/DiscoNet/AT/disco/epoch_100.pth --detection_method none --attack config/attack/multi_agent/collaborative/N02_E5e-02_S10.yaml  --log --logpath experiments/AT_E1e-01_S1/collaborative/N02_E5e-02_S10

# python test_attack_det.py --resume /DB/data/yanghengzhao-1/adversarial/DiscoNet/AT/disco/epoch_100.pth --detection_method none --attack config/attack/multi_agent/non-collaborative/N02_E5e-02_S10.yaml  --log --logpath experiments/AT_E1e-01_S1/non-collaborative/N02_E5e-02_S10

##########################
# keep sparse and positive
##########################

# python test_attack_det.py --detection_method none --attack config/attack/multi_agent/collaborative/keep_sparse_pos/N02_E5e-01_S20_sparse_pos.yaml --log --logpath experiments/attack/keep_sparse_pos/collaborative/N02_E5e-01_S20_sparse_pos

# python test_attack_det.py --detection_method none --attack config/attack/multi_agent/non-collaborative/keep_sparse_pos/N02_E5e-01_S20_sparse_pos.yaml --log --logpath experiments/attack/keep_sparse_pos/non-collaborative/N02_E5e-01_S20_sparse_pos

# python test_attack_det.py --detection_method none --attack config/attack/multi_agent/collaborative/keep_sparse_pos/N03_E5e-01_S20_sparse_pos.yaml --log --logpath experiments/attack/keep_sparse_pos/collaborative/N03_E5e-01_S20_sparse_pos

# python test_attack_det.py --detection_method none --attack config/attack/multi_agent/non-collaborative/keep_sparse_pos/N03_E5e-01_S20_sparse_pos.yaml --log --logpath experiments/attack/keep_sparse_pos/non-collaborative/N03_E5e-01_S20_sparse_pos

###########################
# different attack settings
###########################