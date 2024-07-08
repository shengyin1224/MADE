python train_codet_at.py \
    -d ../v2x-sim-1.0/train \
    --bound lowerbound \
    --com disco \
    --kd_flag 1 \
    --resume_teacher checkpoints/teacher.pth \
    --batch 8 \
    --attack config/attack/single_agent/keep_sparse_pos/N01_E5e-01_S20_sparse_pos.yaml \
    --log \
    --logpath experiments/AT_E5-01_S20_sparse_pos