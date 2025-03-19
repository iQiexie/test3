# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Path, Input
import os
import time
import subprocess
import torch
import urllib.parse
from diffusers import FluxPipeline
from typing import List

from diffusers import FluxTransformer2DModel
from safetensors.torch import load_file

MODEL_CACHE = "./FLUX.1-dev"
MODEL_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
CHECKPOINT_DIR = "./checkpoints"


def download_weights(url, dest, file=False):
    start = time.time()
    print(f"Downloading from {url} to {dest}")
    if not file:
        subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    else:
        subprocess.check_call(["pget", url, dest], close_fds=False)
    print(f"Download completed in {time.time() - start:.2f} seconds")


def download_checkpoint(url, filename=None):
    """Download a checkpoint file from a URL"""
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    filename = f"checkpoint_{int(time.time())}.safetensors"
    checkpoint_path = os.path.join(CHECKPOINT_DIR, filename)

    if not os.path.exists(checkpoint_path):
        print(f"Downloading checkpoint to {checkpoint_path}")
        download_weights(url, checkpoint_path, file=True)
    else:
        print(f"Checkpoint already exists at {checkpoint_path}")

    return checkpoint_path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Download model weights if they don't exist
        if not os.path.exists(MODEL_CACHE):
            print(f"Model weights not found in {MODEL_CACHE}, downloading...")
            download_weights(MODEL_URL, '.')
        else:
            print(f"Model weights found in {MODEL_CACHE}, skipping download")

        # Create checkpoints directory if it doesn't exist
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        # Load the model from the local directory and move it to GPU
        self.pipe = FluxPipeline.from_pretrained(MODEL_CACHE, torch_dtype=torch.bfloat16).to("cuda")

        print("Model loaded successfully on GPU")

    def predict(
        self,
        prompt: str = Input(description="Prompt for the model", default="a tiny astronaut hatching from an egg on the moon"),
        checkpoint_url: str = Input(description="URL to a .safetensors checkpoint file (optional)", default=None),
        adapter_name: str = Input(description="Name for the adapter (used when loading multiple checkpoints)", default="lora"),
        adapter_scale: float = Input(description="Scale for the adapter weights", default=1.0, ge=0.0, le=2.0),
        guidance_scale: float = Input(description="Guidance scale for the diffusion process", default=3.5, ge=0.0, le=20.0),
        height: int = Input(description="Height of the generated image", default=1024, ge=256, le=2048),
        width: int = Input(description="Width of the generated image", default=1024, ge=256, le=2048),
        num_inference_steps: int = Input(description="Number of inference steps", default=8, ge=1, le=100),
        max_sequence_length: int = Input(description="Maximum sequence length", default=512, ge=256, le=1024),
        seed: int = Input(description="Random seed for reproducibility", default=0),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        # Download and load checkpoint if provided
        if checkpoint_url:
            checkpoint_path = download_checkpoint(checkpoint_url)
            print(f"Loading checkpoint from {checkpoint_path}")

            # checkpoint = load_file(checkpoint_path)
            transformer = FluxTransformer2DModel.from_single_file(checkpoint_path, torch_dtype=torch.bfloat16).to("cuda")
            self.pipe = FluxPipeline.from_pretrained(MODEL_CACHE, torch_dtype=torch.bfloat16, transformer=transformer).to("cuda")

            # Unload any existing LoRA weights
            # if hasattr(self.pipe, 'unload_lora_weights'):
            #     self.pipe.unload_lora_weights()

            # # Load the checkpoint as a LoRA adapter
            # self.pipe.load_lora_weights(checkpoint_path, adapter_name=adapter_name)
            # self.pipe.set_adapters([adapter_name], adapter_weights=[adapter_scale])
            # print(f"Checkpoint loaded with adapter_name={adapter_name}, scale={adapter_scale}")

        print(f"Generating image with prompt: '{prompt}'")

        # Set up the generator for reproducibility
        generator = torch.Generator("cuda").manual_seed(seed)

        # Generate the image with the provided parameters
        out = self.pipe(
            prompt=prompt,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            generator=generator,
        ).images[0]

        # Save the output image
        output_path = "flux_output.png"
        out.save(output_path)
        print(f"Image generated and saved as {output_path}")

        # Return the path to the generated image
        return [Path(output_path)]
