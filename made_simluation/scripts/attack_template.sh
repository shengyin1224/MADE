 python test_attack.py \
    -d ../v2x-sim-1.0/test/ \
    --bound lowerbound \
    --com disco \
    --resume checkpoints/DiscoNet.pth \
    --attack TRUE \
    --attack_no_proj \
    --attack_mode others \
   #  --eps 0.1 \
   #  --alpha 0.1 \
   #  --att_target gt \