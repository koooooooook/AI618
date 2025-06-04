import argparse
import itertools
import logging
import math
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler
from transformers import AutoTokenizer
from tqdm.auto import tqdm

import sys
sys.path.append('./')
from concept_training.diffusers_data_pipeline_xl import CustomDiffusionDataset, collate_fn
from concept_training import retrieve


def freeze_params(params):
    for p in params:
        p.requires_grad = False


def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []
    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]
        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def compute_time_ids(resolution, h, w):
    original_size = (resolution, resolution)
    target_size = (resolution, resolution)
    crops_coords_top_left = (h, w)
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    return torch.tensor([add_time_ids])


def save_embeddings(path, text_encoder_one, text_encoder_two, modifier_tokens, ids_one, ids_two):
    state = {"unet": {}, "modifier_token": {}, "modifier_token_2": {}}
    for tok, id1, id2 in zip(modifier_tokens, ids_one, ids_two):
        emb1 = text_encoder_one.get_input_embeddings().weight[id1].detach().cpu()
        emb2 = text_encoder_two.get_input_embeddings().weight[id2].detach().cpu()
        state["modifier_token"][tok] = emb1
        state["modifier_token_2"][tok] = emb2
    torch.save(state, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--instance_prompt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--class_data_dir", type=str, default=None)
    parser.add_argument("--class_prompt", type=str, default=None)
    parser.add_argument("--with_prior_preservation", action="store_true")
    parser.add_argument("--num_class_images", type=int, default=100)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--modifier_token", type=str, required=True)
    parser.add_argument("--initializer_token", type=str, required=True)
    parser.add_argument("--hflip", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crops_coords_top_left_h", type=int, default=0)
    parser.add_argument("--crops_coords_top_left_w", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                              mixed_precision=args.mixed_precision)
    logging.basicConfig(level=logging.INFO)
    if args.seed is not None:
        set_seed(args.seed)

    concepts_list = [{"instance_prompt": args.instance_prompt,
                      "class_prompt": args.class_prompt,
                      "instance_data_dir": args.instance_data_dir,
                      "class_data_dir": args.class_data_dir}]

    if args.with_prior_preservation and args.class_data_dir and args.class_prompt:
        class_dir = Path(args.class_data_dir)
        class_dir.mkdir(parents=True, exist_ok=True)
        if len(list(class_dir.iterdir())) < args.num_class_images:
            if accelerator.is_main_process:
                retrieve.retrieve(args.class_prompt, class_dir, args.num_class_images)
            accelerator.wait_for_everyone()

    tokenizer_one = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2", use_fast=False)
    text_encoder_one = StableDiffusionXLPipeline.from_pretrained(args.pretrained_model_name_or_path).text_encoder
    text_encoder_two = StableDiffusionXLPipeline.from_pretrained(args.pretrained_model_name_or_path).text_encoder_2
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    vae.requires_grad_(False)
    unet.requires_grad_(False)

    modifier_tokens = args.modifier_token.split('+')
    initializer_tokens = args.initializer_token.split('+')
    ids_one = []
    ids_two = []
    for mod, init in zip(modifier_tokens, initializer_tokens[:len(modifier_tokens)]):
        tokenizer_one.add_tokens(mod)
        tokenizer_two.add_tokens(mod)
        token_ids_one = tokenizer_one.encode([init], add_special_tokens=False)
        token_ids_two = tokenizer_two.encode([init], add_special_tokens=False)
        if len(token_ids_one) > 1:
            raise ValueError("initializer token must be one token")
        ids_one.append(tokenizer_one.convert_tokens_to_ids(mod))
        ids_two.append(tokenizer_two.convert_tokens_to_ids(mod))
        text_encoder_one.resize_token_embeddings(len(tokenizer_one))
        text_encoder_two.resize_token_embeddings(len(tokenizer_two))
        token_embeds_one = text_encoder_one.get_input_embeddings().weight.data
        token_embeds_two = text_encoder_two.get_input_embeddings().weight.data
        token_embeds_one[ids_one[-1]] = token_embeds_one[token_ids_one[0]]
        token_embeds_two[ids_two[-1]] = token_embeds_two[token_ids_two[0]]

    params_to_freeze = itertools.chain(
        text_encoder_one.text_model.encoder.parameters(),
        text_encoder_one.text_model.final_layer_norm.parameters(),
        text_encoder_one.text_model.embeddings.position_embedding.parameters(),
        text_encoder_two.text_model.encoder.parameters(),
        text_encoder_two.text_model.final_layer_norm.parameters(),
        text_encoder_two.text_model.embeddings.position_embedding.parameters(),
    )
    freeze_params(params_to_freeze)

    optimizer = torch.optim.AdamW(
        itertools.chain(text_encoder_one.get_input_embeddings().parameters(),
                         text_encoder_two.get_input_embeddings().parameters()),
        lr=args.learning_rate)

    train_dataset = CustomDiffusionDataset(concepts_list=concepts_list,
                                           tokenizer_one=tokenizer_one,
                                           tokenizer_two=tokenizer_two,
                                           with_prior_preservation=args.with_prior_preservation,
                                           size=args.resolution,
                                           center_crop=False,
                                           num_class_images=args.num_class_images,
                                           hflip=args.hflip)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=0,
    )

    weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32

    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=torch.float32)

    add_time_ids = compute_time_ids(args.resolution, args.crops_coords_top_left_h, args.crops_coords_top_left_w)
    if args.with_prior_preservation:
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
    train_dataloader = accelerator.prepare(train_dataloader)
    text_encoder_one, text_encoder_two, optimizer = accelerator.prepare(text_encoder_one, text_encoder_two, optimizer)
    unet = accelerator.prepare(unet)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.max_train_steps
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps * args.gradient_accumulation_steps,
    )

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    global_step = 0

    for epoch in range(1000000):
        text_encoder_one.train()
        text_encoder_two.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(text_encoder_one):
                latents = vae.encode(batch["pixel_values"].to(dtype=vae.dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                latents = latents.to(weight_dtype)
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                unet_added = {"time_ids": add_time_ids.repeat(bsz, 1)}
                prompt_embeds, pooled = encode_prompt([text_encoder_one, text_encoder_two], None, None, [batch["input_ids_one"], batch["input_ids_two"]])
                unet_added.update({"text_embeds": pooled.repeat(bsz, 1)})
                model_pred = unet(noisy_latents, timesteps, prompt_embeds, added_cond_kwargs=unet_added).sample
                target = noise if noise_scheduler.config.prediction_type == "epsilon" else noise_scheduler.get_velocity(latents, noise, timesteps)
                mask = batch["mask"]
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = ((loss * mask).sum([1,2,3]) / mask.sum([1,2,3])).mean()
                accelerator.backward(loss)
                grads_one = text_encoder_one.get_input_embeddings().weight.grad
                grads_two = text_encoder_two.get_input_embeddings().weight.grad
                index_grads_to_zero_one = torch.arange(len(tokenizer_one)) != ids_one[0]
                index_grads_to_zero_two = torch.arange(len(tokenizer_two)) != ids_two[0]
                for idx in ids_one[1:]:
                    index_grads_to_zero_one = index_grads_to_zero_one & (torch.arange(len(tokenizer_one)) != idx)
                for idx in ids_two[1:]:
                    index_grads_to_zero_two = index_grads_to_zero_two & (torch.arange(len(tokenizer_two)) != idx)
                grads_one.data[index_grads_to_zero_one, :] = 0
                grads_two.data[index_grads_to_zero_two, :] = 0
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"delta-{global_step}.bin")
                    save_embeddings(save_path, accelerator.unwrap_model(text_encoder_one), accelerator.unwrap_model(text_encoder_two), modifier_tokens, ids_one, ids_two)
            if global_step >= max_train_steps:
                break
        if global_step >= max_train_steps:
            break
    accelerator.end_training()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, f"delta-{global_step}.bin")
        save_embeddings(save_path, accelerator.unwrap_model(text_encoder_one), accelerator.unwrap_model(text_encoder_two), modifier_tokens, ids_one, ids_two)


if __name__ == "__main__":
    main()
