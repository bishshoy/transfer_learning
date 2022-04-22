# python -m debugpy --listen 5678 --wait-for-client main.py \

# for i in 2e-2 4e-2 6e-2 8e-2; do
# echo "*** VALUE: $i ***"
# done

# OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=4 main.py \
python main.py \
--model vgg16_bn \
--epochs 200 \
--lr 1e-2 \
--cifar 100 \
--replace-fc \

# OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=4 main.py \
python main.py \
--model vgg16_bn \
--pretrained \
--epochs 200 \
--lr 1e-2 \
--cifar 100 \
--freeze-conv \
--replace-fc \


# OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=4 main.py \
python main.py \
--model vgg16_bn \
--pretrained \
--epochs 200 \
--lr 1e-2 \
--cifar 100 \
--replace-fc \

