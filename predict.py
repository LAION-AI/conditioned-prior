import tempfile
import numpy as np
import datetime
import json
import typing

import clip
import torch
from cog import BaseModel, BasePredictor, Path, Input

from util import load_prior, slugify

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

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        prompt: str = Input(description="Text to visualize", default=""),
        candidates: int = Input(
            description="Numer of image embeds to draw from in the prior. Increasing may improve performance.",
            default=2,
            ge=2,
            le=32,
        ),
        cond_scale: float = Input(
            description="How much prior guidance to use.", default=1.0, ge=0.0, le=5.0
        ),
    ) -> Path:
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

        print(f"Tokenizing text: {prompt}")
        text_tokens = clip.tokenize([prompt], truncate=True).to(DEVICE)
        print(f"Encoding text: {prompt}")
        image_embed = self.diffusion_prior.sample(
            text=text_tokens,
            num_samples_per_batch=candidates,
            cond_scale=cond_scale,
        )
        np_image_embed = image_embed.cpu().detach().numpy().astype("float32")[0] # reminder: dont use json, "bad for floats"
        clean_prompt = slugify(prompt)[:50]
        np_save_path = f"image_embed-{clean_prompt}-bs_{candidates}-gs_{cond_scale}.npy"
        np.save(np_save_path, np_image_embed)
        return Path(np_save_path)