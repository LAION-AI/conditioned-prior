import tempfile
import numpy as np
import datetime
import json
import typing

import clip
import torch
from cog import BaseModel, BasePredictor, Path, Input, File

from util import load_prior, slugify

class Output(BaseModel):
    prompt: str
    num_candidates: int
    cond_scale: float
    image_embed: Path
    text_embed: Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Predictor(BasePredictor):
    @torch.inference_mode()
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        prior_model_path = "prior_ema_fp16.pth"
        self.diffusion_prior = (
            load_prior(prior_model_path).to(DEVICE).eval().requires_grad_(False)
        )
        print("loaded diffusion prior")
        self.base_dir = Path(tempfile.mkdtemp())

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        prompt: str = Input(description="Caption to invert to a CLIP image embed", default=""),
        candidates: int = Input(
            description="Numer of image embeds to draw from in the prior. Increasing may improve performance.",
            default=2,
            ge=2,
            le=32,
        ),
        cond_scale: float = Input(
            description="How much prior guidance to use.", default=1.0, ge=0.0, le=5.0
        ),
    ) -> Output:
        """
        Load the model into memory to make running multiple predictions efficient

        Args:
            text_input: Text to visualize.
            prior_guidance_scale: How much prior guidance to use.
            prior_batch_size: Numer of image embeds to generate per batch. Must be greater than 2 so that scoring can be done.
            target_batch_size: Numer of image embeds to generate from the text embed.
            num_results: The number of results to return from the clip-retrieval API.
        """

        assert len(prompt) > 0, "Text input must be non-empty"
        print(f"Predicting CLIP ViT-L-14 image embed from prompt: {prompt}")
        text_tokens = clip.tokenize([prompt], truncate=True).to(DEVICE)

        text_embed = self.diffusion_prior.clip.clip.encode_text(text_tokens)
        text_embed /= text_embed.norm(dim=-1, keepdim=True)
        text_embed = text_embed.cpu().detach().numpy().astype("float32")[0]

        image_embed = self.diffusion_prior.sample(text=text_tokens, num_samples_per_batch=candidates, cond_scale=cond_scale)
        image_embed /= image_embed.norm(dim=-1, keepdim=True)
        image_embed = image_embed.cpu().detach().numpy().astype("float32")[0]

        image_embed_json = image_embed.tolist()
        text_embed_json = text_embed.tolist()

        image_embed_path = self.base_dir.joinpath("image_embed.json")
        text_embed_path = self.base_dir.joinpath("text_embed.json")

        json.dump(image_embed_json, open(image_embed_path, "w"))
        json.dump(text_embed_json, open(text_embed_path, "w"))

        return Output(
            prompt=prompt,
            num_candidates=candidates,
            cond_scale=cond_scale,
            image_embed=image_embed_path,
            text_embed=text_embed_path,
        )