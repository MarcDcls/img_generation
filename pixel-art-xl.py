from diffusers import DiffusionPipeline, LCMScheduler
import torch

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
lcm_lora_id = "latent-consistency/lcm-lora-sdxl"

pipe = DiffusionPipeline.from_pretrained(model_id, variant="fp16")
pipe.enable_xformers_memory_efficient_attention()
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights(lcm_lora_id, adapter_name="lora")
pipe.load_lora_weights("./pixel-art-xl.safetensors", adapter_name="pixel")

pipe.set_adapters(["lora", "pixel"], adapter_weights=[1.0, 1.2])
pipe.to(device="cuda", dtype=torch.float16)

prompt = "pixel, a cute corgi"
negative_prompt = "3d render, realistic"

num_images = 4

for i in range(num_images):
    img = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=8,
        guidance_scale=1.5,
    ).images[0]
    
    img.save(f"lcm_lora_{i}.png")
