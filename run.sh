# python -m debugpy --listen 5678 --wait-for-client experiment.py \

# for i in 'cifar10' 'cifar100' 'stl10' 'caltech101' 'dtd' 'cars' 'aircraft'; do
# for i in 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1; do
# echo "*** VALUE: $i ***"
# done


export MODEL='vgg16_bn'
export DATASET='cifar100'


python experiment.py \
--model $MODEL \
--epochs 100 \
--batch-size 256 \
--lr 1e-2 \
--dataset $DATASET \
--root /datasets \
--replace-fc \
--amp \
