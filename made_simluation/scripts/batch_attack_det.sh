ALPHAS=(2e-2 5e-2 0.1 1e-2)
# for ALPHA in 5e-2
for ALPHA in ${ALPHAS[@]}
do
   for STEP in 1 5 10 15
   do
      echo "no_proj_alpha_${ALPHA}_step_${STEP}"
      python test_attack_det.py \
         -d ../v2x-sim-1.0/test/ \
         --bound lowerbound \
         --com disco \
         --resume checkpoints/DiscoNet.pth \
         --attack TRUE \
         --attack_no_proj \
         --attack_mode others \
         --eps 0.1 \
         --step $STEP \
         --alpha $ALPHA \
         --att_target gt \
         --save_path experiments/no_proj_alpha_${ALPHA}_step_${STEP}/match_costs \
         --log \
         --logpath experiments/no_proj_alpha_${ALPHA}_step_${STEP}/
      python result_analysis.py experiments/no_proj_alpha_${ALPHA}_step_${STEP}/ #--vis --vis_att # --tofile 
   done
done