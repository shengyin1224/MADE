python test_attack_det.py \
   -d ../v2x-sim-1.0/test/ \
   --bound lowerbound \
   --com disco \
   --resume checkpoints/DiscoNet.pth \
   --detection_method binary_classifier \
   --log \
   --logpath experiments/binary_classifier/no_attack/ 

python test_attack_det.py \
   -d ../v2x-sim-1.0/test/ \
   --bound lowerbound \
   --com disco \
   --resume checkpoints/DiscoNet.pth \
   --attack config/attack/multi_agent/collaborative/N02_E1e-01_S10.yaml \
   --detection_method binary_classifier \
   --log \
   --logpath experiments/binary_classifier/collaborative/N01_E1e-01_S10

python test_attack_det.py \
   -d ../v2x-sim-1.0/test/ \
   --bound lowerbound \
   --com disco \
   --resume checkpoints/DiscoNet.pth \
   --attack config/attack/multi_agent/non-collaborative/N02_E1e-01_S10.yaml \
   --detection_method binary_classifier \
   --log \
   --logpath experiments/binary_classifier/non-collaborative/N01_E1e-01_S10

python test_attack_det.py \
   -d ../v2x-sim-1.0/test/ \
   --bound lowerbound \
   --com disco \
   --resume checkpoints/DiscoNet.pth \
   --attack config/attack/single_agent/N01_E1e-01_S10.yaml \
   --detection_method binary_classifier \
   --log \
   --logpath experiments/binary_classifier/single/N01_E1e-01_S10