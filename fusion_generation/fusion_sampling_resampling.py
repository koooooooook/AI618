import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import argparse
from PIL import Image
from tqdm import tqdm
from transformers import logging
from diffusers import DDIMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel,AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
import gc
from utils_custom import *
from sentence_transformers.util import (semantic_search, 
                                        dot_score, 
                                        normalize_embeddings)
import math
import torch.nn.functional as F
from torchvision.utils import save_image

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

logging.set_verbosity_error()

def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

def compute_time_ids():
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
    original_size = (opt.resolution_h, opt.resolution_w)
    target_size = (opt.resolution_h, opt.resolution_w)
    crops_coords_top_left = (opt.crops_coords_top_left_h, opt.crops_coords_top_left_w)
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids])
    # add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
    return add_time_ids

def _gaussian_kernel2d(sigma: float, device):
    k = int(6 * sigma + 1) | 1              # 홀수 kernel size
    ax = torch.arange(k, device=device) - k // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, k, k)

def gaussian_blur(mask: torch.Tensor, sigma: float):
    if sigma <= 0:
        return mask
    kernel = _gaussian_kernel2d(sigma, mask.device).to(mask.dtype)
    pad = kernel.shape[-1] // 2
    mask = F.pad(mask, [pad] * 4, mode='reflect')
    return F.conv2d(mask, kernel)

def preprocess_mask(mask_path, h, w, device, mask_blur_sigma):
    # ── ① PIL → NumPy → Torch (한 번만 차원 추가)
    mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)   # (1,1,H,W)
    mask = (mask > 0.5).float()
    mask = F.interpolate(mask, size=(h, w), mode="nearest")
    return mask

def preprocess_mask_raw(mask_path, h, w, device):
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.nn.functional.interpolate(mask, size=(h, w), mode='nearest')
    return mask

class Tweediemix(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        sd_version = config.sd_version

        if sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        elif sd_version =='1.4':
            model_key = "CompVis/stable-diffusion-v1-4"
        elif sd_version =='xl':
            model_key= "stabilityai/stable-diffusion-xl-base-1.0"
        else:
            raise ValueError(f'Stable-diffusion version {sd_version} not supported.')
        
        # Create SD models
        print('Loading SD model')

        pipe = StableDiffusionXLPipeline.from_pretrained(model_key, torch_dtype=torch.float16, variant="fp16",use_safetensors=True).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()

        pipe.enable_vae_slicing()
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.text_encoder = pipe.text_encoder
        self.text_encoder_2 = pipe.text_encoder_2
        self.unet = pipe.unet
        self.unet.enable_xformers_memory_efficient_attention()
        self.device = self.unet.device
        
        self.sts = []
        self.masks=None

        model_paths = config.personal_checkpoint.split('+')
        prompt_orig = config.prompt_orig.split('+')[0]

        prompt_sep =config.prompt.split('+')
        concepts = config.concepts.split('+')

        modifier_token_user = config.modifier_token.split('+')
        prompts = []
        prompts.append(prompt_orig)
        concept_num = len(concepts)
        prompts_single = prompt_sep[:concept_num-1]
        self.concept_num = concept_num
        for i,wd in enumerate(concepts):
            index = prompt_sep[i].find(wd)
            result = prompt_sep[i][:index] + modifier_token_user[i] + " "+ prompt_sep[i][index:]
            prompts.append(result)
        
        for sp in model_paths:
            self.sts.append(torch.load(sp))
             
        if 'modifier_token' in self.sts[0]:
            modifier_tokens = []
            modifier_tokens_2=[]

            for single_st in self.sts:
                modifier_tokens += list(single_st['modifier_token'].keys())
                modifier_tokens_2 += list(single_st['modifier_token_2'].keys())
                
                
            modifier_token_id = []
            modifier_token_id_2 = []

            # for i,modifier_token in enumerate(modifier_tokens):
            for i,modifier_token in enumerate(modifier_token_user):
                # self.find_disc(self.sts[i]['modifier_token'][modifier_token],self.sts[i]['modifier_token_2'][modifier_token])
                num_added_tokens = self.tokenizer.add_tokens(modifier_token)
                modifier_token_id.append(self.tokenizer.convert_tokens_to_ids(modifier_token))
            # for modifier_token in modifier_tokens_2:
                num_added_tokens = self.tokenizer_2.add_tokens(modifier_token)
                modifier_token_id_2.append(self.tokenizer_2.convert_tokens_to_ids(modifier_token))
                
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))
            self.text_encoder_2.resize_token_embeddings(len(self.tokenizer_2))
            token_embeds = self.text_encoder.get_input_embeddings().weight.data
            token_embeds_2 = self.text_encoder_2.get_input_embeddings().weight.data

            for i, id_ in enumerate(modifier_token_id):
                single_st = self.sts[i]
                token_embeds[id_] = single_st['modifier_token'][modifier_tokens[i]]
            for i, id_ in enumerate(modifier_token_id_2):
                single_st = self.sts[i]
                token_embeds_2[id_] = single_st['modifier_token_2'][modifier_tokens_2[i]]
                
        null_prompt = [config.negative_prompt]

        self.text_embeds = self.get_text_embeds(prompts, null_prompt,device=self.unet.device)

        self.text_embeds_single = self.get_text_embeds(prompts_single,null_prompt,device=self.unet.device)
        
        del self.tokenizer, self.tokenizer_2, self.text_encoder,self.text_encoder_2
        del pipe.tokenizer, pipe.tokenizer_2, pipe.text_encoder, pipe.text_encoder_2, pipe.unet,pipe.vae
        gc.collect()
        torch.cuda.empty_cache()
        
        for i,single_st in enumerate(self.sts):
            setattr(self,f"unet_{i}",UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", torch_dtype=torch.float16, variant="fp16").to(self.device))
            model_name = f"unet_{i}"
            for name, params in getattr(self, model_name).named_parameters():
                if 'attn2' in name:
                    if name in single_st['unet']:
                        params.data.copy_(single_st['unet'][f'{name}'])
            getattr(self, model_name).enable_xformers_memory_efficient_attention()
        
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        N_ts = len(self.scheduler.timesteps)
        self.scheduler.set_timesteps(config.n_timesteps, device=self.unet.device)
        
        self.skip = N_ts // config.n_timesteps
        self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(self.unet.device)
        self.scheduler.alphas_cumprod = torch.cat([torch.tensor([1.0]), self.scheduler.alphas_cumprod])

        print('custom checkpoint loaded')

        self.add_time_ids = compute_time_ids()
        self.add_time_ids = self.add_time_ids.to(self.unet.device)


    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                FusedAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)
    
    def find_disc(self,embed,embed2):
        with torch.no_grad():
            token_embedding = self.text_encoder.get_input_embeddings()
            token_embedding2 = self.text_encoder_2.get_input_embeddings()

            embedding_matrix = token_embedding.weight
            embedding_matrix2 = token_embedding2.weight

            embed = embed.unsqueeze(0)
            embed2 = embed2.unsqueeze(0)
            hits = semantic_search(embed, embedding_matrix.float(), 
                                query_chunk_size=1, 
                                top_k=1,
                                score_function=dot_score)
            hits2 = semantic_search(embed2, embedding_matrix2.float(), 
                                query_chunk_size=1, 
                                top_k=1,
                                score_function=dot_score)

            nn_indices = torch.tensor([hit[0]["corpus_id"] for hit in hits], device=embed.device)
            nn_indices2 = torch.tensor([hit[0]["corpus_id"] for hit in hits2], device=embed.device)
            
    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, device="cuda"):        
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    text_encoders=[self.text_encoder, self.text_encoder_2],
                    tokenizers=[self.tokenizer,self.tokenizer_2],
                    prompt=prompt,
                    text_input_ids_list=None
                )
        uncond_embeds, pooled_uncond_embeds = encode_prompt(
                    text_encoders=[self.text_encoder, self.text_encoder_2],
                    tokenizers=[self.tokenizer,self.tokenizer_2],
                    prompt=negative_prompt,
                    text_input_ids_list=None
                )
        text_embeddings = torch.cat([uncond_embeds,prompt_embeds])
        pooled_text_embeddings = torch.cat([pooled_uncond_embeds,pooled_prompt_embeds])
        return text_embeddings,pooled_text_embeddings
    
    def prepare_extra_step_kwargs(self, generator, eta):

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    @torch.no_grad()
    def decode_latent(self, latent):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latent = 1 / 0.18215 * latent
            img = self.vae.decode(latent).sample
            img = (img / 2 + 0.5).clamp(0, 1)
        return img
    
    def alpha(self, t):
        at = self.scheduler.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod
        return at

    def slerp_multiple_noises_(self, noise, num_noises):
        """
        Spherical linear interpolation across multiple noise tensors.
        
        Args:
            noise (torch.Tensor): noise tensor of shape [B, ...]
            num_noises (int): number of noise tensors to interpolate

        Returns:
            torch.Tensor: interpolated noise tensor of shape [B, ...]
        """
        noises = [noise]
        
        for i in range(num_noises):
            eps = torch.randn_like(noise) * self.scheduler.init_noise_sigma
            noises.append(eps)

        # Flatten all noises for interpolation
        flat_noises = [n.view(n.shape[0], -1) for n in noises]
        interpolated = flat_noises[0]  # start from the first
        for i in range(1, len(flat_noises)):
            n_i = flat_noises[i]
            alpha = 1.0 / (i + 1)  # ← 고정된, 단순 역수

            dot = (interpolated * n_i).sum(dim=1, keepdim=True)
            dot = dot / (interpolated.norm(dim=1, keepdim=True) * n_i.norm(dim=1, keepdim=True)).clamp(min=1e-8)
            dot = dot.clamp(-1.0, 1.0)

            theta = torch.acos(dot)
            sin_theta = torch.sin(theta).clamp(min=1e-6)

            interpolated = (
                torch.sin((1 - alpha) * theta) * interpolated +
                torch.sin(alpha * theta) * n_i
            ) / sin_theta

        return interpolated.view_as(noises[0])

    @torch.no_grad()
    def denoise_step(self, x, t):

        text_embed_cond,text_embed_cond_pool=self.text_embeds

        next_t = t - self.skip
        at = self.alpha(t)
        at_next = self.alpha(next_t)

        sizes = x.shape
        # if self.masks is not None:
        #     register_time(self, t.item(),self.masks)
        # else:
        register_time(self, t.item())        

        if t<=self.t_cond_cur:
            text_embed_uncond = text_embed_cond[0].unsqueeze(0)
            text_embed_concept = text_embed_cond[2:]

            text_embed_uncond_pool = text_embed_cond_pool[0].unsqueeze(0)
            text_embed_concept_pool = text_embed_cond_pool[2:]

            latent_model_input = torch.cat([x]*(self.concept_num+1))
            text_embed = torch.cat([ text_embed_uncond,
                                    text_embed_concept], dim=0)
        
            text_embed_pool = torch.cat([text_embed_uncond_pool,
                                    text_embed_concept_pool], dim=0)

            unet_added_conditions = {"time_ids": self.add_time_ids.repeat(text_embed.shape[0], 1)}
            unet_added_conditions.update({"text_embeds": text_embed_pool})
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed,added_cond_kwargs=unet_added_conditions)['sample']
        else:
            text_embed_uncond = text_embed_cond[0].unsqueeze(0)
            text_embed_multi = text_embed_cond[1].unsqueeze(0)

            text_embed_uncond_pool = text_embed_cond_pool[0].unsqueeze(0)
            text_embed_multi_pool = text_embed_cond_pool[1].unsqueeze(0)
            if t==self.start_t:
                text_embed_cond_single,text_embed_cond_pool_single=self.text_embeds_single
                text_embed_cond_single = text_embed_cond_single[1:]
                text_embed_cond_pool_single = text_embed_cond_pool_single[1:]

                latent_model_input = torch.cat([x]*(self.concept_num+1))

                text_embed = torch.cat([ text_embed_uncond
                                ,text_embed_multi,
                                text_embed_cond_single], dim=0)
                text_embed_pool = torch.cat([text_embed_uncond_pool
                                    ,text_embed_multi_pool,
                                    text_embed_cond_pool_single, ], dim=0)
            else:

                latent_model_input = torch.cat([x]+[x])
                text_embed = torch.cat([ text_embed_uncond
                                    ,text_embed_multi], dim=0)
                text_embed_pool = torch.cat([text_embed_uncond_pool
                                    ,text_embed_multi_pool], dim=0)

            unet_added_conditions = {"time_ids": self.add_time_ids.repeat(text_embed.shape[0], 1)}
            unet_added_conditions.update({"text_embeds": text_embed_pool})

            unet_added_conditions_single = {"time_ids": self.add_time_ids.repeat(text_embed[:2].shape[0], 1)}
            unet_added_conditions_single.update({"text_embeds": text_embed_pool[:2]})
            
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed,added_cond_kwargs=unet_added_conditions)['sample']

        noise_pred_uncond = noise_pred[:1]

        if t<=self.t_cond_cur:
            # noise_preds = []
            denoised_tweedie = 0
            for cc in range(self.concept_num):
                noise_pred_cond = noise_pred[(1+cc):(2+cc)]
                noise_pred_concept = noise_pred_uncond + self.config.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                
                denoised_tweedie += self.masks[cc].unsqueeze(0)*((x - (1-at).sqrt() * noise_pred_concept) / at.sqrt())
        else:
            if t==self.start_t:
                if self.config.resampling_steps>0:
                    print(f'resampling {self.config.resampling_steps} steps')
                    for _ in range(self.config.resampling_steps):
                        noise_pred_uncond = noise_pred[:1]
                        noise_pred_mult = noise_pred[1:2]
                        noise_pred_mult = noise_pred_uncond + self.config.guidance_scale * (noise_pred_mult - noise_pred_uncond)
                        denoised_tweedie_mult = (x - (1-at).sqrt() * noise_pred_mult) / at.sqrt()
                        denoised_tweedie = (self.concept_num-1)*denoised_tweedie_mult
                        
                        for cc in range(self.concept_num-1):
                            noise_pred_single = noise_pred_uncond + self.config.guidance_scale * (noise_pred[2+cc:3+cc] - noise_pred_uncond)
                            denoised_tweedie_single = (x - (1-at).sqrt() * noise_pred_single) / at.sqrt()
                            denoised_tweedie -= denoised_tweedie_single
                            
                        denoised_latent = at_next.sqrt() * denoised_tweedie + (1-at_next).sqrt() * noise_pred_uncond
                        latent_model_next = torch.cat([denoised_latent]+[denoised_latent])
                        
                        noise_pred_next = self.unet(latent_model_next, next_t, encoder_hidden_states=text_embed[:2],added_cond_kwargs=unet_added_conditions_single)['sample']
                        noise_pred_cond_next = noise_pred_next[1:2]
                        noise_pred_uncond_next = noise_pred_next[:1]
                        noise_pred_next = noise_pred_uncond_next + self.config.guidance_scale * (noise_pred_cond_next - noise_pred_uncond_next)     

                        denoised_tweedie_next = (denoised_latent - (1-at_next).sqrt() * noise_pred_next) / at_next.sqrt()
                        return_x = at.sqrt() * denoised_tweedie_next + (1-at).sqrt() * noise_pred_uncond_next
                        latent_model_input = torch.cat([return_x]*(self.concept_num+1))   
                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed,added_cond_kwargs=unet_added_conditions)['sample']
                        x = return_x
                # else
                del noise_pred_next, noise_pred_cond_next, noise_pred_uncond_next,denoised_tweedie_next,latent_model_next
                gc.collect()
                torch.cuda.empty_cache()

                noise_pred_cond = noise_pred[1:2]
                noise_pred_uncond = noise_pred[:1]
                noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred_cond = noise_pred[1:2]
                noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_cond - noise_pred_uncond)

            denoised_tweedie = (x - (1-at).sqrt() * noise_pred) / at.sqrt()
        
        denoised_latent = at_next.sqrt() * denoised_tweedie + (1-at_next).sqrt() * noise_pred_uncond
        
        if t ==self.t_cond_prev:
            denoised_latent_temp = denoised_latent
            t_temp = next_t
            if self.config.jumping_steps>0:
                for _ in range(self.config.jumping_steps):
                    at_temp = self.alpha(t_temp)
                    
                    latent_model_next = torch.cat([denoised_latent_temp]+[denoised_latent_temp])
                    noise_pred_next = self.unet(latent_model_next, t_temp, encoder_hidden_states=text_embed[:2],added_cond_kwargs=unet_added_conditions_single)['sample']
                    noise_pred_cond_next = noise_pred_next[1:2]
                    noise_pred_uncond_next = noise_pred_next[:1]
                    noise_pred_next = noise_pred_uncond_next + self.config.guidance_scale * (noise_pred_cond_next - noise_pred_uncond_next)     
                    t_temp = t_temp - 150
                    at_temp_next = self.alpha(t_temp)
                    denoised_tweedie = (denoised_latent_temp - (1-at_temp).sqrt() * noise_pred_next) / at_temp.sqrt()
                    denoised_latent_temp = at_temp_next.sqrt() * denoised_tweedie + (1-at_temp_next).sqrt() * noise_pred_uncond_next
                
                del noise_pred_next, noise_pred_cond_next, noise_pred_uncond_next,denoised_latent_temp,latent_model_next
                gc.collect()
                torch.cuda.empty_cache()

                decoded_tweedie = self.decode_latent(denoised_tweedie)
            else:
                decoded_tweedie = self.decode_latent(denoised_tweedie)
            path_tweedie = os.path.join( self.config.output_path ,f'tweedie.jpg')
            T.ToPILImage()(decoded_tweedie[0]).save(path_tweedie)
            test_cmd = f'CUDA_VISIBLE_DEVICES={self.config.seg_gpu} python text_segment/run_expand.py --input_path={path_tweedie} --text_condition="{self.config.seg_concepts}" --output_path={self.config.output_path} --mask_type={self.config.mask_type}'
            os.system(test_cmd)

            mask_paths = []
            concept_list = self.config.seg_concepts.split('+')
            for sp in concept_list:
                mask_paths.append(os.path.join(self.config.output_path,sp+'.jpg'))
            
            # MASKS
            fg_masks = torch.cat([preprocess_mask(mask_path, self.config.resolution_h // 8, self.config.resolution_w // 8, self.unet.device, self.config.mask_blur_sigma) for mask_path in mask_paths])
            for i in range(len(mask_paths)):
                mask_path = mask_paths[i]
                hard_path = os.path.splitext(mask_path)[0] + "_hard.png"
                save_image(fg_masks[i].cpu(), hard_path)
            
            # MASK NORMALIZATION
            if self.config.mask_blur_sigma > 0:
                fg_masks = gaussian_blur(fg_masks, self.config.mask_blur_sigma).clamp_(0, 1)                
                for i in range(len(mask_paths)):
                    mask_path = mask_paths[i]
                    soft_path = os.path.splitext(mask_path)[0] + "_soft.png"
                    save_image(fg_masks[i].cpu(), soft_path)
                
            # BACKGROUND MASK    
            bg_mask = 1 - torch.sum(fg_masks, dim=0, keepdim=True).clamp_(0, 1)
            hard_path = os.path.join(self.config.output_path , "background_hard.png")
            save_image(bg_mask.cpu(), hard_path)
            if self.config.normalize_masks:
                bg_mask = gaussian_blur(bg_mask, self.config.mask_blur_sigma).clamp_(0, 1)
                soft_path = os.path.join(self.config.output_path , "background_soft.png")
                save_image(bg_mask.cpu(), soft_path)
            
            # NORMALIZE MASKS
            if self.config.normalize_masks:
                print("### Normalizing masks ###")                
                masks_all = torch.cat([fg_masks, bg_mask], dim=0)
                sum_all = masks_all.sum(dim=0, keepdim=True).clamp(min=1e-4)
                masks_norm = masks_all / sum_all
                fg_masks = masks_norm[:-1]
                for i in range(len(mask_paths)):
                    mask_path = mask_paths[i]
                    soft_path = os.path.splitext(mask_path)[0] + "_normalized.png"
                    save_image(fg_masks[i].cpu(), soft_path)
                bg_mask  = masks_norm[-1:]
                soft_path = os.path.join(self.config.output_path , "background_normalized.png")
                save_image(bg_mask.cpu(), soft_path)
                
            self.masks = torch.cat([fg_masks, bg_mask])
        
        if t ==1:
            denoised_latent = denoised_tweedie
        
        return denoised_latent

    @torch.no_grad()
    def global_refine(self, x0: torch.Tensor):
        """
        샘플링이 끝난 latent x0를 self.unet으로 4–8 DDIM 스텝만 살짝 리터치.
        * encoder_hidden_states 2행(uncond+multi)로 전달 → 컨셉별 to_k_i 사용 안 함
        """
        
        # DDIM timesteps: 마지막 s+1개만 사용 (σ가 매우 작은 구간)
        ddim = DDIMScheduler.from_config(self.scheduler.config)
        start_t = int(50 * self.config.global_refine_frac)
        start_t = max(start_t, self.config.global_refine_steps + 2)                           # 최소 보장
        ddim.set_timesteps(start_t + 1, device=self.device)     # 0…start_t
        ts = ddim.timesteps[-(self.config.global_refine_steps + 1):]                          # start_t 근처 σ 구간
        ddim.timesteps = ts.flip(0)[:-1]                        # s개만 남김

        # x0 → x_t0 (아주 작은 노이즈 추가)
        t0 = ts[-1]
        alpha0 = ddim.alphas_cumprod[t0]
        noise = torch.randn_like(x0) * self.config.global_refine_sigma
        x = alpha0.sqrt() * x0 + (1 - alpha0).sqrt() * noise

        # 두 줄짜리 임베딩(uncond + multi)만 준비
        text_embed_cond, text_embed_pool_cond = self.text_embeds
        text_embed        = text_embed_cond[:2]         # [uncond, multi]
        text_embed_pool   = text_embed_pool_cond[:2]    # 동일 길이

        # DDIM 루프
        print(f'Global refine: {self.config.global_refine_steps} steps')
        for t in ddim.timesteps:
            latent_model_input = torch.cat([x, x])      # (2, C, H, W)

            added_cond = {
                "time_ids": self.add_time_ids.repeat(2, 1),
                "text_embeds": text_embed_pool,
            }

            eps = self.unet(latent_model_input, t,
                            encoder_hidden_states=text_embed,
                            added_cond_kwargs=added_cond).sample
            eps_uncond = eps[0:1]
            eps_cond   = eps[1:2]

            g  = self.config.global_refine_guidance
            eps_final = eps_uncond + g * (eps_cond - eps_uncond)

            x = ddim.step(eps_final, t, x, eta=0).prev_sample

        return x

    def init_fusion(self, t_cond):
        self.t_cond = self.scheduler.timesteps[t_cond:] if t_cond >= 0 else []
        self.t_cond_prev = self.scheduler.timesteps[t_cond-1]
        self.t_cond_cur = self.scheduler.timesteps[t_cond]
        self.start_t = self.scheduler.timesteps[0]
        os.makedirs(self.config.output_path_all, exist_ok=True)
        register_attention_control_efficient(self, self.t_cond, self.concept_num)
        for i in range(self.concept_num): delattr(self, f'unet_{i}')
        
    def run_fusion(self):
        t_cond = int(self.config.n_timesteps * self.config.t_cond)
        self.init_fusion(t_cond=t_cond)
        normal = torch.randn(1,4,self.config.resolution_h//8,self.config.resolution_w//8).to(self.unet.device)  * self.scheduler.init_noise_sigma
        if self.config.use_slerp_noise >= 2:
            normal = self.slerp_multiple_noises_(normal, self.config.use_slerp_noise)
        _ = self.sample_loop(normal)
        
    @torch.no_grad()
    def sample_loop(self, x):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):
                x = self.denoise_step(x, t)
            
            if self.config.global_refine_steps is not None and self.config.global_refine_steps != 0:
                x = self.global_refine(x)
            
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                x = x.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            elif x.dtype != self.vae.dtype:
                if torch.backends.mps.is_available():
                    self.vae = self.vae.to(x.dtype)

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(x.device, x.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(x.device, x.dtype)
                )
                x = x * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                x = x / self.vae.config.scaling_factor

            decoded_latent = self.vae.decode(x, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            
            os.makedirs(self.config.output_path_all, exist_ok=True)
            image = self.image_processor.postprocess(decoded_latent, output_type='pil')
            image[0].save(f'{self.config.output_path_all}/{self.config.prompt_orig}_{self.config.seed}.png')
            image[0].save(f'./test_out_imgs/{self.config.seed}_{self.config.filename_postfix}.png')
            print("################################################################################\n")
                
        return decoded_latent


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=182)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--output_path_all', type=str)
    parser.add_argument('--negative_prompt', type=str, default='blurry, ugly, black, low res, unrealistic, blurry face')
    # 'blurry, ugly, black, low res, unrealistic, blurry face'
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.4','1.5', '2.0','2.1','xl'],
                        help="stable diffusion version")
    parser.add_argument('--t_cond', type=float, default=0.4)
    parser.add_argument('--guidance_scale', type=float, default=9.0)
    parser.add_argument('--n_timesteps', type=int, default=50)
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--prompt_orig', type=str, default='')
    parser.add_argument('--seg_concepts', type=str, default='')                           
    parser.add_argument('--personal_checkpoint', type=str, default='')
    parser.add_argument('--concepts', type=str)
    parser.add_argument('--modifier_token', type=str)
    parser.add_argument('--resampling_steps',  type=int, default=10)
    parser.add_argument('--jumping_steps',  type=int, default=5)
    parser.add_argument('--seg_gpu',  type=int, default=1)
    parser.add_argument('--use_slerp_noise', type=int, default=1,
                        help='use SLERP-mixed initial noise')
    parser.add_argument('--mask_blur_sigma', type=float, default=0,
                        help="Gaussian σ for mask soft-blurring (0 = no blur)")
    parser.add_argument('--mask_type', type=str, default='rectangular', choices=['rectangular', 'detail'],)
    parser.add_argument('--global_refine_steps',   type=int,
                        help='샘플링 종료 뒤 추가 DDIM 스텝 수')
    parser.add_argument('--global_refine_guidance',type=float,
                        help='추가 리터치 단계 CFG 스케일')
    parser.add_argument('--global_refine_sigma',   type=float,
                        help='리터치 시작 시 삽입할 노이즈 표준편차')
    parser.add_argument('--global_refine_frac', type=float,
                        help='리터치 시작 시점: 0.0~1.0 사이의 비율로, 0.5면 전체 DDIM 스텝의 절반에서 시작')
    parser.add_argument('--normalize_masks', action='store_true',
                        help='마스크를 정규화합니다. (0~1 범위로 스케일링)')
    parser.add_argument('--filename_postfix', type=str)
    parser.add_argument(
        "--crops_coords_top_left_h",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--crops_coords_top_left_w",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--resolution_h",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--resolution_w",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    opt = parser.parse_args()
    
    seed_everything(opt.seed)
    tweedie = Tweediemix(opt)
    tweedie.run_fusion()