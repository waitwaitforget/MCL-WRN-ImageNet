import os

# imagenet data folder: /home/ktian/data/ILSVRC2012/
os.system('python ensemble_multiprocess.py --dataroot /home/ktian/data/ILSVRC2012/ \
                                           --batch-size 128 \
                                           -k 1 \
                                           --arch resnet18 \
                                           --model-num 3 \
                                           --exp-name test_imagenet_mcl \
                                           --debug \
                                           --gpu 1,2,3 \
                                           --cuda \
                                           --ensemble mcl')
