# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Path
import torch
from diffusers import FluxPipeline
from typing import List

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Load the model from the local directory
        self.pipe = FluxPipeline.from_pretrained("./FLUX.1-dev", torch_dtype=torch.bfloat16)
        
        # Apply memory optimizations
        self.pipe.enable_sequential_cpu_offload()  # More aggressive memory optimization
        self.pipe.vae.enable_slicing()  # Enable VAE slicing
        self.pipe.vae.enable_tiling()  # Enable VAE tiling

    def predict(self) -> List[Path]:
        """Run a single prediction on the model"""
        # Define the prompt
        prompt = "a tiny astronaut hatching from an egg on the moon"
        
        # Generate the image with reduced resolution and fewer steps
        out = self.pipe(
            prompt=prompt,
            guidance_scale=3.5,
            height=512,  # Reduced height
            width=768,   # Reduced width
            num_inference_steps=20,  # Fewer steps
        ).images[0]
        
        # Save the output image
        output_path = "flux_output.png"
        out.save(output_path)
        print(f"Image generated and saved as {output_path}")
        
        # Return the path to the generated image
        return [Path(output_path)]