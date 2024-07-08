# ALPHAS=(2e-2 5e-2 0.1 1e-2)
EPSILONS=(5e-2 1e-1 5e-1)
ALPHAS=(1e-2 2e-2 1e-1)
# for ALPHA in 5e-2
# for ALPHA in ${ALPHAS[@]}
# for EPS in ${EPSILONS[@]}
for ((i=0; i<3; ++i))
do
   STEP=10
   echo "pgd_alpha_${ALPHAS[$i]}_step_${STEP}_eps_${EPSILONS[$i]}"
   # mv experiments/pgd_alpha__step_${STEP}_eps_${EPSILONS[$i]} experiments/pgd_alpha_${ALPHAS[$i]}_step_${STEP}_eps_${EPSILONS[$i]}
   python test_attack_det.py \
      -d ../v2x-sim-1.0/test/ \
      --bound lowerbound \
      --com disco \
      --resume checkpoints/DiscoNet.pth \
      --attack TRUE \
      --attack_mode others \
      --eps 0.1 \
      --step ${STEP} \
      --alpha ${ALPHAS[$i]} \
      --eps ${EPSILONS[$i]} \
      --att_target gt \
      --save_path experiments/pgd_alpha_${ALPHAS[$i]}_step_${STEP}_eps_${EPSILONS[$i]}/match_costs \
      --log \
      --logpath experiments/pgd_alpha_${ALPHAS[$i]}_step_${STEP}_eps_${EPSILONS[$i]}/
   python result_analysis.py experiments/pgd_alpha_${ALPHAS[$i]}_step_${STEP}_eps_${EPSILONS[$i]}/ --vis --tofile 
done