python -u main_stage2.py \
--gpus 0 \
--model resnet18_1w1a \
--results_dir ./result/stage2 \
--dataset cifar10 \
--epochs 300 \
--lr 0.0037 \
-b 256 \
-bt 128 \
--lr_type cos \
--warm_up \
--weight_decay 0.00016 \
--tau 0.0468 \
--load_ckpt_stage1 ./result/stage1/model_best.pth.tar \
# --resume