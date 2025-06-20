export PYTHONWARNINGS="ignore"

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export CUDA_VISIBLE_DEVICES=0

## GPU for text-guided segmentation 
## we recommend to use different GPU from generation to avoid memory issue
SEG_GPU=1

## background must comes last
## prompt for concept 1 + prompt for concept 2 + prompt for concept 3(background)
PROMPT="photo of a cat running, mountain background+photo of a dog running, mountain background+mountain background"
## prompt for multiple concepts 
PROMPT_ORIG="photo of a cat and a dog running, mountain background"

SEED=3817
NUM_NOISES=3
EXP_CODE="final_catdog"
export RESULT_PATH="./test_out/resampling_${EXP_CODE}"

## concept order must be the same as the prompt
CONCEPTS="cat+dog+mountain"
MODIFIER="<cat1>+<dog1>+<mountain1>"
## concepts for text-guided segmentation. background concept must not be included
SEG_CONCEPTS="a cat+a dog"

## guidance_scale = CFG weight 0<=guidance_scale<=1
## t_cond = timestep to start multiconcept sampling
## jumping_steps = number of steps to sampling from intermediate tweedie
## resampling_steps = number of steps to multi-concept resampling

## -----------when using custom diffusion weights-----------

PERSONAL_CHECKPOINT="./checkpoint_custom/cat1.bin+./checkpoint_custom/dog1.bin+./checkpoint_custom/mountain1.bin"

python fusion_generation/fusion_sampling_resampling.py \
--guidance_scale 0.8 --n_timesteps 50 --prompt "$PROMPT" --personal_checkpoint $PERSONAL_CHECKPOINT \
--output_path $RESULT_PATH --output_path_all $RESULT_PATH --sd_version "xl" --concepts "$CONCEPTS" --modifier_token $MODIFIER --resolution_h 1024 --resolution_w 1024 \
--prompt_orig "$PROMPT_ORIG" --seed $SEED --t_cond 0.2 --seg_concepts="$SEG_CONCEPTS" --negative_prompt '' --seg_gpu $SEG_GPU \
--use_slerp_noise $NUM_NOISES --mask_type "rectangular" --mask_blur_sigma 4.0 --normalize_masks \
--filename_postfix "${EXP_CODE}"


## -----------when using lora weights-----------

# PERSONAL_CHECKPOINT="../checkpoint_custom/pet_cat1_lora/delta-1000.bin+../checkpoint_custom/pet_dog1_lora/delta-1000.bin+../checkpoint_custom/wululu_lora/delta-1000.bin"

# python fusion_generation/fusion_sampling_lora.py \
# --guidance_scale 0.8 --n_timesteps 50 --prompt "$PROMPT" --personal_checkpoint $PERSONAL_CHECKPOINT \
# --output_path $RESULT_PATH --output_path_all $RESULT_PATH --sd_version "xl" --concepts "$CONCEPTS" --modifier_token $MODIFIER --resolution_h 1024 --resolution_w 1024 \
# --prompt_orig "$PROMPT_ORIG" --seed $SEED --t_cond 0.2 --seg_concepts="$SEG_CONCEPTS" --negative_prompt '' --seg_gpu $SEG_GPU