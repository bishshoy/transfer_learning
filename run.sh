# python -m debugpy --listen 5678 --wait-for-client experiment.py \

# for i in 'cifar10' 'cifar100' 'stl10' 'caltech101' 'dtd' 'cars' 'aircraft'; do
# for i in 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1; do
# echo "*** VALUE: $i ***"
# done


export MODEL='resnet18'
# export MODEL='vgg16'
# export MODEL='vgg16_bn'
# export MODEL='densenet121'
# export MODEL='inceptionv3'
# export MODEL='mobilenetv2'

python experiment.py \
--model $MODEL \
--pretrained \
--epochs 1 \
--batch-size 128 \
--lr 1e-2 \
--dataset dtd \

