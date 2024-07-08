# no attack
CUDA_VISIBLE_DEVICES=0 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --dataset test --temperature 20


# pgd
CUDA_VISIBLE_DEVICES=7 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --attack config/0224_attack/pgd/test_attack.yaml --dataset test --temperature 20

# match
CUDA_VISIBLE_DEVICES=1 python inference.py \
    --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --attack config/0224_attack/pgd/test_iou.yaml --dataset test --temperature 20 \
    --standard match \
    --range_dataset [20,30]

CUDA_VISIBLE_DEVICES=1 python inference.py \
    --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --dataset test --temperature 20 \
    --standard match

# robosac
CUDA_VISIBLE_DEVICES=1 python inference.py \
    --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --attack config/0224_attack/pgd/test_attack.yaml --dataset test --temperature 20 \
    --standard robosac \
    --robosac_cfg config/defense/robosac.yml \
    --range_dataset [20,30]

CUDA_VISIBLE_DEVICES=0 python inference.py \
    --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --dataset test --temperature 20 \
    --standard robosac \
    --robosac_cfg config/defense/robosac.yml

# robosac-probing
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --attack config/0224_attack/pgd/test_attack.yaml --dataset test --temperature 20 \
    --standard robosac \
    --robosac_cfg config/defense/robosac_probing.yml

#############
# 0310

# 1. attack
CUDA_VISIBLE_DEVICES=0 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --attack config/0224_attack/pgd/test_attack.yaml --dataset test --temperature 20

# 2. match
CUDA_VISIBLE_DEVICES=1 python inference.py \
    --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --attack config/0224_attack/pgd/test_iou.yaml --dataset test --temperature 20 \
    --standard match

# 3. robosac
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --attack config/0224_attack/pgd/test_attack.yaml --dataset test --temperature 20 \
    --standard robosac \
    --robosac_cfg config/defense/robosac_probing.yml


#############

# 0311

# 测试eps大的时候ae是否可以区分攻击和正常样本
CUDA_VISIBLE_DEVICES=0 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --attack config/0224_attack/pgd/test_attack_eps5.yml --dataset test --temperature 20 --range_dataset [20,100]

CUDA_VISIBLE_DEVICES=0 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --attack config/0224_attack/pgd/test_attack_eps5_temp5.yml --dataset test --temperature 5 --range_dataset [20,100]

CUDA_VISIBLE_DEVICES=0 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --attack config/0224_attack/pgd/eps10.yml --dataset test --temperature 1 --range_dataset [20,100]

# 生成validation set
CUDA_VISIBLE_DEVICES=0 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --dataset validation --temperature 20 --standard g_ae_val_validation_1016_unet_double --ae_type residual

# attack eps 1.5
CUDA_VISIBLE_DEVICES=0 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --dataset test --temperature 20 --standard ae --ae_type residual \
    --attack config/0224_attack/pgd/test_attack.yaml --range_dataset [20,100]

# attack eps 5 测试
CUDA_VISIBLE_DEVICES=0 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --dataset test --temperature 20 --standard ae --ae_type residual \
    --attack config/0224_attack/pgd/test_attack_eps5.yml --range_dataset [20,100]

CUDA_VISIBLE_DEVICES=0 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --dataset test --temperature 20 --standard ae --ae_type residual \
    --attack config/0224_attack/pgd/test_attack_eps5_temp5.yml --range_dataset [20,100]

CUDA_VISIBLE_DEVICES=0 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --dataset test --temperature 20 --standard ae --ae_type residual \
    --attack config/0224_attack/pgd/eps10.yml --range_dataset [20,100]

# no attack测试
CUDA_VISIBLE_DEVICES=0 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --dataset test --temperature 20 --standard ae --ae_type residual \
    --range_dataset [20,100]


########### 0312

# 1. attack
CUDA_VISIBLE_DEVICES=0 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --attack config/0224_attack/pgd/test_attack_eps1.yml --dataset test --temperature 20

CUDA_VISIBLE_DEVICES=1 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --attack config/0224_attack/pgd/test_attack_eps2.yml --dataset test --temperature 20

# 2. match
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --attack config/0224_attack/pgd/test_attack_eps1.yml --dataset test --temperature 20 \
    --standard match

CUDA_VISIBLE_DEVICES=1 python inference.py \
    --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --attack config/0224_attack/pgd/test_attack_eps2.yml --dataset test --temperature 20 \
    --standard match

# 3. res ae
CUDA_VISIBLE_DEVICES=0 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --dataset test --temperature 20 --standard ae --ae_type residual \
    --attack config/0224_attack/pgd/test_attack_eps1.yml

CUDA_VISIBLE_DEVICES=0 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --dataset test --temperature 20 --standard ae --ae_type residual \
    --attack config/0224_attack/pgd/test_attack.yml

CUDA_VISIBLE_DEVICES=1 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --dataset test --temperature 20 --standard ae --ae_type residual \
    --attack config/0224_attack/pgd/test_attack_eps2.yml

# 4. multi test
CUDA_VISIBLE_DEVICES=0 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --dataset test --temperature 20 --standard multi_test \
    --attack config/0224_attack/pgd/test_attack_eps1.yml

CUDA_VISIBLE_DEVICES=1 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --dataset test --temperature 20 --standard multi_test \
    --attack config/0224_attack/pgd/test_attack_eps2.yml


#### ALL ROBOSAC

CUDA_VISIBLE_DEVICES=0 python inference.py \
    --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --attack config/0224_attack/pgd/test_attack.yml --dataset test --temperature 20 \
    --standard robosac \
    --robosac_cfg config/defense/robosac.yml

CUDA_VISIBLE_DEVICES=0 python inference.py \
    --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --attack config/0224_attack/pgd/test_attack_eps1.yml --dataset test --temperature 20 \
    --standard robosac \
    --robosac_cfg config/defense/robosac.yml

CUDA_VISIBLE_DEVICES=1 python inference.py \
    --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --attack config/0224_attack/pgd/test_attack_eps2.yml --dataset test --temperature 20 \
    --standard robosac \
    --robosac_cfg config/defense/robosac.yml

CUDA_VISIBLE_DEVICES=1 python inference.py \
    --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --dataset test --temperature 20 \
    --standard robosac \
    --robosac_cfg config/defense/robosac.yml


##############################################

## 0314

# 1. generate_validation_set

CUDA_VISIBLE_DEVICES=0 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --dataset validation --temperature 20 --standard g_match_validation_1016_para_0.01 --match_para 0.01

CUDA_VISIBLE_DEVICES=0 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --dataset validation --temperature 20 --standard g_match_validation_1016_para_0.5 --match_para 0.5

CUDA_VISIBLE_DEVICES=1 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --dataset validation --temperature 20 --standard g_match_validation_1016_para_2 --match_para 2

CUDA_VISIBLE_DEVICES=1 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --dataset validation --temperature 20 --standard g_match_validation_1016_para_100 --match_para 100


# 2. test match cost

CUDA_VISIBLE_DEVICES=0 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --dataset test --temperature 20 --standard match --match_para 0.01

CUDA_VISIBLE_DEVICES=0 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --dataset test --temperature 20 --standard match --match_para 0.5

CUDA_VISIBLE_DEVICES=1 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --dataset test --temperature 20 --standard match --match_para 2

CUDA_VISIBLE_DEVICES=1 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --dataset test --temperature 20 --standard match --match_para 100


CUDA_VISIBLE_DEVICES=0 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --dataset test --temperature 20 --standard match --match_para 0.01 --attack config/0224_attack/pgd/test_attack.yml

CUDA_VISIBLE_DEVICES=0 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --dataset test --temperature 20 --standard match --match_para 0.5 --attack config/0224_attack/pgd/test_attack.yml

CUDA_VISIBLE_DEVICES=1 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --dataset test --temperature 20 --standard match --match_para 2 --attack config/0224_attack/pgd/test_attack.yml

CUDA_VISIBLE_DEVICES=1 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --dataset test --temperature 20 --standard match --match_para 100 --attack config/0224_attack/pgd/test_attack.yml


# 3. multi test

CUDA_VISIBLE_DEVICES=0 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --dataset test --temperature 20 --standard multi_test --attack config/0224_attack/pgd/test_attack.yml --multi_test_alpha 0.01

CUDA_VISIBLE_DEVICES=1 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --dataset test --temperature 20 --standard multi_test --attack config/0224_attack/pgd/test_attack.yml --multi_test_alpha 0.1


CUDA_VISIBLE_DEVICES=0 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --dataset test --temperature 20 --standard multi_test --multi_test_alpha 0.01

CUDA_VISIBLE_DEVICES=1 python inference.py --model_dir logs/dair_centerpoint_multiscale_att_from_0 --fusion_method intermediate \
    --dataset test --temperature 20 --standard multi_test --multi_test_alpha 0.1


cp -r box_overlaps.egg-info build config dist docs logs outcome Rotated_IoU rubbish sort_vertices.egg-info v2xsim_vistool inference.py num_agent_list.npy requirements.txt rubbish.sh train_index.npy trainset_idx.json validation_index.npy /dssg/home/acct-seecsh/seecsh/shengyin/DAMC-HPC