
# for i in 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1; do
# echo "*** VALUE: $i ***"
# done
# python -m debugpy --listen 5678 --wait-for-client main.py \



# python main.py \
# --model vgg16_bn \
# --epochs 200 \
# --lr 1e-2 \
# --cifar 10 \
# --replace-fc \


# python main.py \
# --model vgg16_bn \
# --pretrained \
# --epochs 200 \
# --lr 1e-2 \
# --cifar 10 \
# --freeze-conv \
# --replace-fc \


python main.py \
--model vgg16_bn \
--pretrained \
--epochs 200 \
--lr 1e-2 \
--cifar 10 \
--replace-fc \

