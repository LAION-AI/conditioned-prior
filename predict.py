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

def save_results_to_json_file(text_input, results):
    current_time_as_path_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    clean_prompt = slugify(text_input)
    with open(f"results_{current_time_as_path_str}_{clean_prompt}.json", "w") as f:
        json.dump(results, f)
        print(f"Wrote results to {f.name}")
        return f.name

class Predictor(BasePredictor):
    @torch.inference_mode()
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        prior_model_path = "prior_L_fp16.pth"
        self.diffusion_prior = (
            load_prior(prior_model_path).to(DEVICE).eval().requires_grad_(False)
        )
        print("loaded diffusion prior")

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        text_input: str = Input(description="Text to visualize", default=""),
        prior_batch_size: int = Input(
            description="Numer of image embeds to draw from in the prior. Increasing may improve performance.",
            default=2,
            ge=2,
            le=32,
        ),
        prior_guidance_scale: float = Input(
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
        assert len(text_input) > 0, "Text input must be non-empty"

        print(f"Tokenizing text: {text_input}")
        text_tokens = clip.tokenize([text_input], truncate=True).to(DEVICE)
        print(f"Encoding text: {text_input}")
        image_embed = self.diffusion_prior.sample(
            text=text_tokens,
            num_samples_per_batch=prior_batch_size,
            cond_scale=prior_guidance_scale,
        )
        np_image_embed = image_embed.cpu().detach().numpy().astype("float32")[0] # reminder: dont use json, "bad for floats"
        clean_prompt = slugify(text_input)[:50]
        np_save_path = f"image_embed-{clean_prompt}-bs_{prior_batch_size}-gs_{prior_guidance_scale}.npy"
        np.save(np_save_path, np_image_embed)
        return Path(np_save_path)
