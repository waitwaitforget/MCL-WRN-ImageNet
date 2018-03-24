import os

os.system('python ensemble_multiprocess.py --dataroot ./data \
                                           --batch-size 128 \
                                           -k 1 \
                                           --arch test \
                                           --model-num 3 \
                                           --exp-name test_mnist \
                                           --debug \
                                           --gpu 0\
                                           --ensemble mcl')
