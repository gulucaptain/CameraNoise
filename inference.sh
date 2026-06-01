VGGT_MODEL="checkpoints/VGGT-1B"
CAMERANOISE_CONFIG="cameranoise_warping/configs/default.yaml"
QWENVL_MODEL="checkpoints/Qwen2-VL-7B-Instruct"
WAN_MODEL="checkpoints/Wan2.1-I2V-14B-720P"
CAMERANOISE_LORA_MODEL="checkpoints/CameraNoise-I2V/cameranoise_lora.safetensors"

for i in {1..10}; do
    DEMO_DIR="outputs/demo${i}"

    if [ ! -d "$DEMO_DIR" ]; then
        echo "Skip ${DEMO_DIR}: directory not found."
        continue
    fi

    echo "========================================"
    echo "Running ${DEMO_DIR}"
    echo "========================================"

    python cameranoise_i2v.py \
        --demo-dir "$DEMO_DIR" \
        --vggt-ckpt "$VGGT_MODEL" \
        --cameranoise-config "$CAMERANOISE_CONFIG" \
        --qwenvl-model-path "$QWENVL_MODEL" \
        --model-root "$WAN_MODEL" \
        --lora-path "$CAMERANOISE_LORA_MODEL" \
        --height 576 \
        --width 1024 \
        --frames 49 \
        --sample-mode front \
        --degradation-value 0.2 \
        --cfg 3.5 \
        --device cuda \
        --output-type single
done
