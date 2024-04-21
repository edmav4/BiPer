# Example for BiPer-ResNet18 model
python -u main.py \
--gpus 0 \
--model resnet18_1w1a \
--results_dir ./result/stage1 \
--dataset cifar10 \
--epochs 600 \
--lr 0.021 \
-b 256 \
-bt 128 \
--lr_type cos \
--warm_up \
--weight_decay 0.0016 \
--tau 0.037 \
--freq 20
# --resume