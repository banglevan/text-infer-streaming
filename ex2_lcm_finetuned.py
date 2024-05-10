from diffusers import DiffusionPipeline, LCMScheduler
import torch

pipe = DiffusionPipeline.from_pretrained(
    "Linaqruf/animagine-xl",
    variant="fp16",
    torch_dtype=torch.float16
).to("cuda")

# set scheduler
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# load LCM-LoRA
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")

prompt = "face focus, cute, masterpiece, best quality, 1girl, green hair, sweater, looking at viewer, upper body, beanie, outdoors, night, turtleneck"

generator = torch.manual_seed(0)
import time
for i in range(5):
    tic = time.time()
    image = pipe(
        prompt=prompt, num_inference_steps=4, generator=generator, guidance_scale=1.0
    ).images[0]
    toc = time.time()
    print(f'{toc - tic: .4f}')

from matplotlib import pyplot as plt

plt.imshow(image)
plt.show()