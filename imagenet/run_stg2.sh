python main_stage2.py \
--gpus 0 \
--model resnet18_1w1a \
--data_path data \
--dataset imagenet \
--epochs 100 \
--lr 0.01 \
-b 512 \
-bt 256 \
--lr_type cos \
--weight_decay 1e-4 \
--tau_min 0.0  \
--tau_max 0.0  \
--freq 20 \
--load_ckpt_stage1 ./result/stage1/model_best.pth.tar \
--use_dali \
# --resume