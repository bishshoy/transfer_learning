# python -m debugpy --listen 5678 --wait-for-client experiment.py \

# for i in 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1; do
# echo "*** VALUE: $i ***"
# done


python experiment.py \
--model vgg16_bn \
--epochs 200 \
--batch-size 128 \
--lr 1e-2 \
--replace-fc \
--dataset stl10 \


# python experiment.py \
# --model vgg16_bn \
# --pretrained \
# --epochs 200 \
# --lr $i \
# --cifar 100 \
# --droot /cifar100 \
# --freeze-conv \
# --replace-fc \
# --check-hyp \


# python experiment.py \
# --model vgg16_bn \
# --pretrained \
# --epochs 200 \
# --lr 1e-2 \
# --cifar 100 \
# --droot /cifar100 \
# --replace-fc \

