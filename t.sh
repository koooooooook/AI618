
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export CUDA_VISIBLE_DEVICES=1

## GPU for text-guided segmentation 
## we recommend to use different GPU from generation to avoid memory issue
SEG_GPU=1

## background must comes last
## prompt for concept 1 + prompt for concept 2 + prompt for concept 3(background)
# PROMPT="photo of a cat running, mountain background+photo of a dog running, mountain background+mountain background"
# PROMPT="photo of a <dog1> dog, garden background+photo of a <cat1> cat, garden background+<garden1>garden background"
PROMPT="photo of a <dogcatgarden1> dog and cat+<garden1> garden background"
## prompt for multiple concepts 
# PROMPT_ORIG="photo of a cat and a dog running, mountain background"
PROMPT_ORIG="photo of a <dogcatgarden1> dog and cat, with <garden1> garden background"

export RESULT_PATH="./test_out"
# export SEED=3821
export SEED=3827

## concept order must be the same as the prompt
# CONCEPTS="cat+dog+mountain"
CONCEPTS="dogcatgarden+garden"
# MODIFIER="<cat1>+<dog1>+<mountain1>"
MODIFIER="<dogcatgarden1>+<garden1>"
## concepts for text-guided segmentation. background concept must not be included
# SEG_CONCEPTS="a cat+a dog"
SEG_CONCEPTS="a dog and a cat+a garden"

## guidance_scale = CFG weight 0<=guidance_scale<=1
## t_cond = timestep to start multiconcept sampling
## jumping_steps = number of steps to sampling from intermediate tweedie
## resampling_steps = number of steps to multi-concept resampling

## -----------when using custom diffusion weights-----------

# PERSONAL_CHECKPOINT="./checkpoint_custom/cat1.bin+./checkpoint_custom/dog1.bin+./checkpoint_custom/mountain1.bin"
# PERSONAL_CHECKPOINT="./checkpoint_custom/pet_dog1/delta-200.bin+./checkpoint_custom/pet_cat1/delta-200.bin+./checkpoint_custom/scene_garden/delta-200.bin"
PERSONAL_CHECKPOINT="./checkpoint_custom/td_dogcatgarden/delta-200.bin+./checkpoint_custom/scene_garden/delta-200.bin"

python fusion_generation/fusion_sampling.py \
--guidance_scale 0.8 --n_timesteps 50 --prompt "$PROMPT" --personal_checkpoint $PERSONAL_CHECKPOINT \
--output_path $RESULT_PATH --output_path_all $RESULT_PATH --sd_version "xl" --concepts "$CONCEPTS" --modifier_token $MODIFIER --resolution_h 1024 --resolution_w 1024 \
--prompt_orig "$PROMPT_ORIG" --seed $SEED --t_cond 0.2 --seg_concepts="$SEG_CONCEPTS" --negative_prompt '' --seg_gpu $SEG_GPU


## -----------when using lora weights-----------

# PERSONAL_CHECKPOINT="../checkpoint_custom/pet_cat1_lora/delta-1000.bin+../checkpoint_custom/pet_dog1_lora/delta-1000.bin+../checkpoint_custom/wululu_lora/delta-1000.bin"

# python fusion_generation/fusion_sampling_lora.py \
# --guidance_scale 0.8 --n_timesteps 50 --prompt "$PROMPT" --personal_checkpoint $PERSONAL_CHECKPOINT \
# —output_path $RESULT_PATH —output_path_all $RESULT_PATH —sd_version "xl" —concepts "$CONCEPTS" —modifier_token $MODIFIER —resolution_h 1024 —resolution_w 1024 \
# —prompt_orig "$PROMPT_ORIG" —seed $SEED —t_cond 0.2 —seg_concepts="$SEG_CONCEPTS" —negative_prompt '' —seg_gpu $SEG_GPU