# Time Test

# 1. attack
CUDA_VISIBLE_DEVICES=1 python test_attack.py \
    -d ../v2x-sim-1.0/test/ \
    --bound lowerbound \
    --com disco \
    --resume checkpoints/DiscoNet.pth \
    --attack config/attack/single_agent/N01_E1e-01_S10.yaml

# 2. match
CUDA_VISIBLE_DEVICES=1 python test_attack_det.py --detection_method match_cost_v2 --attack config/attack/single_agent/N01_E1e-01_S10.yaml --log --logpath experiments/match_cost_v2/ --att_subpath N01_E1e-01_S10

# 3. res ae
CUDA_VISIBLE_DEVICES=4 python test_attack_det.py --detection_method residual_autoencoder_v2 --attack config/attack/single_agent/N01_E1e-01_S10.yaml --log --logpath experiments/residual_ae_v2/N01_E1e-01_S10

# 4. multi test
CUDA_VISIBLE_DEVICES=4 python test_attack_det.py --detection_method multi-test --attack config/attack/single_agent/N01_E1e-01_S10.yaml --log --logpath experiments/multi-test-v2/N01_E1e-01_S10


##### 0311 One More Time - Daft Punk

# 1. attack eps=0.05, 0.2

CUDA_VISIBLE_DEVICES=2 python test_attack_det.py --detection_method none --attack config/attack/single_agent/N01_E5e-02_S10.yaml --log --logpath experiments/attack/single/ --att_subpath N01_E5e-02_S10 --save_attack

CUDA_VISIBLE_DEVICES=3 python test_attack_det.py --detection_method none --attack config/attack/single_agent/N01_E2e-01_S10.yaml --log --logpath experiments/attack/single/ --att_subpath N01_E2e-01_S10 --save_attack


# 2. match

CUDA_VISIBLE_DEVICES=2 python test_attack_det.py --detection_method match_cost_v2 --attack config/attack/single_agent/N01_E5e-02_S10.yaml --log --logpath experiments/match_cost_v2/ --att_subpath N01_E5e-02_S10

CUDA_VISIBLE_DEVICES=3 python test_attack_det.py --detection_method match_cost_v2 --attack config/attack/single_agent/N01_E2e-01_S10.yaml --log --logpath experiments/match_cost_v2/ --att_subpath N01_E2e-01_S10

# 3. ae
CUDA_VISIBLE_DEVICES=2 python test_attack_det.py --detection_method residual_autoencoder_v2 --attack config/attack/single_agent/N01_E5e-02_S10.yaml --log --logpath experiments/residual_ae_v2/N01_E5e-02_S10

CUDA_VISIBLE_DEVICES=3 python test_attack_det.py --detection_method residual_autoencoder_v2 --attack config/attack/single_agent/N01_E2e-01_S10.yaml --log --logpath experiments/residual_ae_v2/N01_E2e-01_S10

# 4. multi test
CUDA_VISIBLE_DEVICES=2 python test_attack_det.py --detection_method multi-test-v3 --attack config/attack/single_agent/N01_E5e-02_S10.yaml --log --logpath experiments/multi-test-v3/ --att_subpath N01_E5e-02_S10

CUDA_VISIBLE_DEVICES=3 python test_attack_det.py --detection_method multi-test-v3 --attack config/attack/single_agent/N01_E2e-01_S10.yaml --log --logpath experiments/multi-test-v3/ --att_subpath N01_E2e-01_S10

################ 0312

# 1. robosac
CUDA_VISIBLE_DEVICES=1 python test_attack_det.py --detection_method robosac --log --logpath experiments/robosac/no_attack --robosac_cfg config/robosac.yml

CUDA_VISIBLE_DEVICES=6 python test_attack_det.py --detection_method robosac --attack config/attack/single_agent/N01_E5e-02_S10.yaml --log --logpath experiments/robosac/new_N01_E5e-02_S10 --robosac_cfg config/robosac.yml

CUDA_VISIBLE_DEVICES=1 python test_attack_det.py --detection_method robosac --attack config/attack/single_agent/N01_E2e-01_S10.yaml --log --logpath experiments/robosac/known_N01_E2e-01_S10 --robosac_cfg config/robosac.yml


############### 0314

# 1. match cost
CUDA_VISIBLE_DEVICES=1 python test_attack_det.py --detection_method match_cost_v2 --log --logpath experiments/match_cost_v2/ablation_0.01/no_attack --match_para 0.01

CUDA_VISIBLE_DEVICES=0 python test_attack_det.py --detection_method match_cost_v2 --log --logpath experiments/match_cost_v2/ablation_0.5/no_attack --match_para 0.5

CUDA_VISIBLE_DEVICES=2 python test_attack_det.py --detection_method match_cost_v2 --log --logpath experiments/match_cost_v2/ablation_2/no_attack --match_para 2

CUDA_VISIBLE_DEVICES=2 python test_attack_det.py --detection_method match_cost_v2 --log --logpath experiments/match_cost_v2/ablation_100/no_attack --match_para 100


CUDA_VISIBLE_DEVICES=0 python test_attack_det.py --detection_method match_cost_v2 --log --logpath experiments/match_cost_v2/ablation_0.01 --match_para 0.01 --attack config/attack/single_agent/N01_E1e-01_S10.yaml --att_subpath N01_E1e-01_S10

CUDA_VISIBLE_DEVICES=6 python test_attack_det.py --detection_method match_cost_v2 --log --logpath experiments/match_cost_v2/ablation_0.5 --match_para 0.5 --attack config/attack/single_agent/N01_E1e-01_S10.yaml --att_subpath N01_E1e-01_S10

CUDA_VISIBLE_DEVICES=0 python test_attack_det.py --detection_method match_cost_v2 --log --logpath experiments/match_cost_v2/ablation_2 --match_para 2 --attack config/attack/single_agent/N01_E1e-01_S10.yaml --att_subpath N01_E1e-01_S10

CUDA_VISIBLE_DEVICES=2 python test_attack_det.py --detection_method match_cost_v2 --log --logpath experiments/match_cost_v2/ablation_100 --match_para 100 --attack config/attack/single_agent/N01_E1e-01_S10.yaml --att_subpath N01_E1e-01_S10

# 2. multi test
CUDA_VISIBLE_DEVICES=2 python test_attack_det.py --detection_method multi-test-v3 --attack config/attack/single_agent/N01_E1e-01_S10.yaml --log --logpath experiments/multi-test-v3/ablation_0.01 --att_subpath N01_E1e-01_S10 --multi_test_alpha 0.01

CUDA_VISIBLE_DEVICES=3 python test_attack_det.py --detection_method multi-test-v3 --attack config/attack/single_agent/N01_E1e-01_S10.yaml --log --logpath experiments/multi-test-v3/ablation_0.1 --att_subpath N01_E1e-01_S10 --multi_test_alpha 0.1


CUDA_VISIBLE_DEVICES=2 python test_attack_det.py --detection_method multi-test-v3 --log --logpath experiments/multi-test-v3/ablation_0.01 --multi_test_alpha 0.01

CUDA_VISIBLE_DEVICES=3 python test_attack_det.py --detection_method multi-test-v3 --log --logpath experiments/multi-test-v3/ablation_0.1 --multi_test_alpha 0.1


#############
# 0315 Bad Mode - Hikaru Utada

# 1. match cost with attack and save match cost

CUDA_VISIBLE_DEVICES=0 python test_attack_det.py --detection_method match_cost_v2 --log --logpath experiments/match_cost_v2/ablation_0.01_0315 --match_para 0.01 --attack config/attack/single_agent/N01_E1e-01_S10.yaml --att_subpath N01_E1e-01_S10 --save_path /GPFS/data/shengyin-1/damc-yanghengzhao/disco-net/match_cost/save_match_cost/ablation_0.01_0315

CUDA_VISIBLE_DEVICES=0 python test_attack_det.py --detection_method match_cost_v2 --log --logpath experiments/match_cost_v2/ablation_0.5_0315 --match_para 0.5 --attack config/attack/single_agent/N01_E1e-01_S10.yaml --att_subpath N01_E1e-01_S10 --save_path /GPFS/data/shengyin-1/damc-yanghengzhao/disco-net/match_cost/save_match_cost/ablation_0.5_0315

CUDA_VISIBLE_DEVICES=3 python test_attack_det.py --detection_method match_cost_v2 --log --logpath experiments/match_cost_v2/ablation_2_0315 --match_para 2 --attack config/attack/single_agent/N01_E1e-01_S10.yaml --att_subpath N01_E1e-01_S10 --save_path /GPFS/data/shengyin-1/damc-yanghengzhao/disco-net/match_cost/save_match_cost/ablation_2_0315

CUDA_VISIBLE_DEVICES=3 python test_attack_det.py --detection_method match_cost_v2 --log --logpath experiments/match_cost_v2/ablation_100_0315 --match_para 100 --attack config/attack/single_agent/N01_E1e-01_S10.yaml --att_subpath N01_E1e-01_S10 --save_path /GPFS/data/shengyin-1/damc-yanghengzhao/disco-net/match_cost/save_match_cost/ablation_100_0315

# 2. match cost without attack
CUDA_VISIBLE_DEVICES=4 python test_attack_det.py --detection_method match_cost_v2 --log --logpath experiments/match_cost_v2/ablation_0.01_0315/no_attack --match_para 0.01

CUDA_VISIBLE_DEVICES=4 python test_attack_det.py --detection_method match_cost_v2 --log --logpath experiments/match_cost_v2/ablation_0.5_0315/no_attack --match_para 0.5

CUDA_VISIBLE_DEVICES=5 python test_attack_det.py --detection_method match_cost_v2 --log --logpath experiments/match_cost_v2/ablation_2_0315/no_attack --match_para 2

CUDA_VISIBLE_DEVICES=7 python test_attack_det.py --detection_method match_cost_v2 --log --logpath experiments/match_cost_v2/ablation_100_0315/no_attack --match_para 100

# 3. multi test no attack

CUDA_VISIBLE_DEVICES=0 python test_attack_det.py --detection_method multi-test-v3 --log --logpath experiments/multi-test-v3/ablation_0.01_no_attack --multi_test_alpha 0.01

CUDA_VISIBLE_DEVICES=2 python test_attack_det.py --detection_method multi-test-v3 --log --logpath experiments/multi-test-v3/ablation_0.1_no_attack --multi_test_alpha 0.1

# 4. feature visualize
CUDA_VISIBLE_DEVICES=5 python test_attack_det.py --detection_method none --attack config/attack/single_agent/N01_E2e-01_S10.yaml --log --logpath experiments/attack/single/ --att_subpath N01_E2e-01_S10


##### no attack

CUDA_VISIBLE_DEVICES=4 python test_attack_det.py --detection_method none --log --logpath experiments/no_attack/single/ --att_subpath no_attack_vis --visualization --save_path /GPFS/data/shengyin-1/damc-yanghengzhao/disco-net/experiments/visualize/no_attack_single

CUDA_VISIBLE_DEVICES=4 python test_attack_det.py --detection_method none --log --logpath experiments/attack/single/ --att_subpath attack_N01_E2e-01_S10_vis --visualization --attack config/attack/single_agent/N01_E2e-01_S10.yaml --save_path /GPFS/data/shengyin/damc-yanghengzhao/disco-net/experiments/visualize/single_agent_attack


#### 0319

CUDA_VISIBLE_DEVICES=0 python test_attack_det.py --detection_method no_com --log --logpath experiments/ego_single/ --att_subpath attack_vis --visualization --save_path /GPFS/data/shengyin-1/damc-yanghengzhao/disco-net/experiments/visualize/ego_single_real --attack config/attack/single_agent/N01_E2e-01_S10.yaml

CUDA_VISIBLE_DEVICES=0 python test_attack_det.py --detection_method none --log --logpath experiments/ego_single/ --att_subpath attack_vis --visualization --save_path /GPFS/data/shengyin-1/damc-yanghengzhao/disco-net/experiments/visualize/no_attack_ego_single

CUDA_VISIBLE_DEVICES=3 python test_attack_det.py --detection_method multi-test-v3 --attack config/attack/single_agent/N01_E2e-01_S10.yaml --log --logpath experiments/multi-test-v3/ --att_subpath N01_E2e-01_S10 --visualization --save_path /GPFS/data/shengyin-1/damc-yanghengzhao/disco-net/experiments/visualize/multi_test_v3_N01_E2e-01_S10


#### 0321

# eps = 0.05

CUDA_VISIBLE_DEVICES=4 python test_attack_det.py --detection_method none --attack config/attack/single_agent/N01_E5e-02_S10.yaml --log --logpath experiments/single/ --att_subpath N01_E5e-02_S10 --visualization --save_path /GPFS/data/shengyin-1/damc-yanghengzhao/disco-net/experiments/visualize/N01_E5e-02_S10_vis

# eps = 0.1
CUDA_VISIBLE_DEVICES=5 python test_attack_det.py --detection_method none --attack config/attack/single_agent/N01_E1e-01_S10.yaml --log --logpath experiments/single/ --att_subpath N01_E1e-01_S10 --visualization --save_path /GPFS/data/shengyin-1/damc-yanghengzhao/disco-net/experiments/visualize/N01_E1e-01_S10_vis

# match
CUDA_VISIBLE_DEVICES=6 python test_attack_det.py --detection_method match_cost_v2 --attack config/attack/single_agent/N01_E2e-01_S10.yaml --log --logpath experiments/match_cost_v2/ --att_subpath N01_E2e-01_S10 --visualization --save_path /GPFS/data/shengyin-1/damc-yanghengzhao/disco-net/experiments/visualize/match_cost_v2_N01_E2e-01_S10

# res ae
CUDA_VISIBLE_DEVICES=7 python test_attack_det.py --detection_method residual_autoencoder_v2 --attack config/attack/single_agent/N01_E2e-01_S10.yaml --log --logpath experiments/residual_ae_v2/ --att_subpath N01_E2e-01_S10 --visualization --save_path /GPFS/data/shengyin-1/damc-yanghengzhao/disco-net/experiments/visualize/residual_ae_v2_N01_E2e-01_S10

# robosac
CUDA_VISIBLE_DEVICES=0 python test_attack_det.py --detection_method robosac --attack config/attack/single_agent/N01_E2e-01_S10.yaml --log --logpath experiments/residual_ae_v2/ --att_subpath N01_E2e-01_S10 --visualization --save_path /GPFS/data/shengyin-1/damc-yanghengzhao/disco-net/experiments/visualize/robosac_N01_E2e-01_S10 --robosac_cfg config/robosac.yml


# oracle
CUDA_VISIBLE_DEVICES=2 python test_attack_det.py --detection_method oracle --attack config/attack/single_agent/N01_E2e-01_S10.yaml --log --logpath experiments/oracle/ --att_subpath N01_E2e-01_S10 --visualization --save_path /GPFS/data/shengyin-1/damc-yanghengzhao/disco-net/experiments/visualize/oracle_N01_E2e-01_S10