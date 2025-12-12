
MODEL="ast"
SEED="1 2 3 4 5"

for s in $SEED
do
    for m in $MODEL
    do
        TAG="bs4_accum16_lr5e-5_ep50_seed${s}_dat2_sex"
        CUDA_VISIBLE_DEVICES=2 python main.py --tag $TAG \
                                        --dataset all \
                                        --seed $s \
                                        --class_split wheeze \
                                        --n_cls 2 \
                                        --epochs 50 \
                                        --batch_size 4 \
                                        --optimizer adam \
                                        --learning_rate 5e-5 \
                                        --weight_decay 1e-6 \
                                        --cosine \
                                        --model $m \
                                        --test_fold official \
                                        --pad_types repeat \
                                        --weighted_loss \
                                        --resz 1 \
                                        --n_mels 128 \
                                        --ma_update \
                                        --ma_beta 0.5 \
                                        --from_sl_official \
                                        --audioset_pretrained \
                                        --method ce \
                                        --domain_adaptation2 \
                                        --meta_mode sex \
                                        --accum \
                                        --accum_steps 16 \
                                        --print_freq 500

                                        # only for evaluation, add the following arguments
                                        # --eval \
                                        # --pretrained \
                                        # --pretrained_ckpt ./save/icbhi_ast_ce_bs8_lr5e-5_ep50_seed1/best.pth

    done
done
