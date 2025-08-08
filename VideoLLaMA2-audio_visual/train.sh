RUN_NAME=audio_visual_stage3_qwen2
DATA_DIR=datasets


python -u videollama2/train_EVQA.py \
    --model_type videollama2_qwen2 \
    --model_path /root/workspace/cvuaggk7v38s73dgjft0/videollama2weights \
    --data_folder ${DATA_DIR} \
    --data_path /root/workspace/cvuaggk7v38s73dgjft0/code/VideoLLaMA2-audio_visual/train.json \
    --vision_tower google/siglip-so400m-patch14-384 \
    --audio_tower /root/workspace/cvuaggk7v38s73dgjft0/videollama2weights/audio_tower.bin \
    --pretrain_mm_mlp_adapter_a /root/workspace/cvuaggk7v38s73dgjft0/videollama2weights/mm_projector_a.bin \
    --mm_projector_type stc_connector_v35 \
    --mm_projector_a_type mlp2x_gelu \
    --va True \
    --tune_audio_tower True \
    --tune_adapter_llm True \
    --tune_mm_mlp_adapter_a True \
    --mm_vision_select_layer -2 \
    --num_frames 8 \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --loss_type mse \
    --output_dir /root/workspace/cvuaggk7v38s73dgjft0/videollama2_EVQA_weights_mse \
    --num_train_epochs 1 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 17 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --run_name $RUN_NAME | tee training_log.txt
