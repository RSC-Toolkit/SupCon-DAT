
MODEL="cnn6"
SEED="1 2 3 4 5"

for s in $SEED
do
    for m in $MODEL
    do
        TAG="bs8_lr5e-5_ep50_seed${s}_multitask_diagnosis_wl"
        CUDA_VISIBLE_DEVICES=3 python main.py --tag $TAG \
                                        --dataset all \
                                        --seed $s \
                                        --class_split multitask \
                                        --multitask \
                                        --lung_cls 2 \
                                        --disease_cls 3 \
                                        --epochs 200 \
                                        --batch_size 64 \
                                        --optimizer adam \
                                        --learning_rate 1e-3 \
                                        --weight_decay 1e-6 \
                                        --cosine \
                                        --model $m \
                                        --test_fold official \
                                        --pad_types repeat \
                                        --weighted_loss_diagnosis_only \
                                        --resz 1 \
                                        --n_mels 128 \
                                        --ma_update \
                                        --ma_beta 0.5 \
                                        --from_sl_official \
                                        --audioset_pretrained \
                                        --method ce \
                                        --print_freq 500

                                        # only for evaluation, add the following arguments
                                        # --eval \
                                        # --pretrained \
                                        # --pretrained_ckpt ./save/icbhi_ast_ce_bs8_lr5e-5_ep50_seed1/best.pth

    done
done
