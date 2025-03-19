import torch
from diffusers import FluxPipeline

# Load the model from the local directory
pipe = FluxPipeline.from_pretrained("./FLUX.1-dev", torch_dtype=torch.bfloat16)

# Apply memory optimizations
pipe.enable_sequential_cpu_offload()  # More aggressive memory optimization
pipe.vae.enable_slicing()  # Enable VAE slicing
pipe.vae.enable_tiling()  # Enable VAE tiling

# Define the prompt
prompt = "a tiny astronaut hatching from an egg on the moon"

# Generate the image with reduced resolution and fewer steps
out = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    height=512,  # Reduced height
    width=768,   # Reduced width
    num_inference_steps=20,  # Fewer steps
).images[0]

# Save the output image
out.save("flux_output.png")
print("Image generated and saved as flux_output.png")