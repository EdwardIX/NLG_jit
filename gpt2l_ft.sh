python src/gpt2_ft.py \
    --train_data ./data/e2e/train.jsonl \
    --valid_data ./data/e2e/valid.jsonl \
    --train_batch_size 1 \
    --grad_acc 8 \
    --valid_batch_size 1 \
    --seq_len 512 \
    --model_card gpt2.lg \
    --init_checkpoint ./pretrained_checkpoints/gpt2-large-pytorch_model.bin \
    --clip 0.0 \
    --lr 0.0002 \
    --weight_decay 0.01 \
    --correct_bias \
    --adam_beta2 0.999 \
    --scheduler linear \
    --warmup_step 4000 \
    --max_epoch 5 \
    --save_interval 1000 \
    --lora_dim 4 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --label_smooth 0.1 \
    --work_dir ./trained_models/GPT2_L_jit/e2e \
    --random_seed 110 \