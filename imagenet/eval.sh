python main_stage2.py \
--gpus 0 \
-e pretrained_models/biper_imagenet_resnet34_stage2/model_best.pth.tar \
--model resnet34_1w1a \
--dataset imagenet \
-bt 256 \

#python main_stage2.py \
#--gpus 0 \
#-e pretrained_models/biper_imagenet_resnet18_stage2/model_best.pth.tar \
#--model resnet18_1w1a \
#--dataset imagenet \
#-bt 256 \

#python main_stage2.py \
#--gpus 0 \
#-e {checkpoint_path} \
#--model {model arch} \
#--dataset imagenet \
#-bt 256 \