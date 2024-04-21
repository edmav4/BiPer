python main.py \
--gpus 0,1,2,3 \
--model resnet18_1w1a \
--data_path data \
--dataset imagenet \
--epochs 200 \
--lr 0.1 \
--weight_decay 1e-4 \
-b 512 \
-bt 256 \
--lr_type cos \
--freq 20 \
--warm_up \
--tau_min 0.85  \
--tau_max 0.99  \
--print_freq 250 \
--use_dali
#--resume