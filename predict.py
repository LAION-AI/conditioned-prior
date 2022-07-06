import tempfile

import clip
import numpy as np
import torch
from cog import BaseModel, BasePredictor, Input, Path
from typing import List

from util import load_prior, slugify


class PriorOutput(BaseModel):
    """
    Pydantic class for specifying the return type of the prior prediction API endpoint.
    """
    text_embedding: List[float]
    image_embedding: List[float]

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
        self.base_dir = Path(".embed_cache")
        self.base_dir.mkdir(exist_ok=True, parents=True)

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        prompt: str = Input(
            description="Caption to invert to a CLIP image embed", default=""
        ),
        candidates: int = Input(
            description="Numer of image embeds to draw from in the prior. Increasing may improve performance.",
            default=2,
            ge=2,
            le=32,
        ),
        cond_scale: float = Input(
            description="How much prior guidance to use.", default=1.0, ge=0.0, le=5.0
        ),
        overwrite: bool = Input(
            description="Re-predict any cached embeddings.", default=False
        )
    ) -> PriorOutput:
        """
        Load the model into memory to make running multiple predictions efficient

        Args:
            prompt: Text to invert to a CLIP image embed
            cond_scale: How much prior guidance to use.
            candidates: Number of image embeds to draw from in the prior. Increasing may improve performance. 
            overwrite: Whether to overwrite the cached embedding if it already exists. (default: False)

        Returns:
            A `PriorOutput` object containing the text tokens, text embed and the image embed.
            * `text_embedding` numpy file containing ndarray of type `float`, included for convenience.
            * `image_embedding` numpy file containing ndarray of type `float`, representing the image embedding predicted by the model.
        """
        # Setup
        assert len(slugify(prompt)) > 0, "Your prompt is either empty or contains only invalid characters."
        print(f"Predicting CLIP ViT-L-14 image embed from prompt: {prompt}")

        image_embed_path = Path(
            self.base_dir, f"image_embed_{slugify(prompt)}.npy"
        )

        # Encode tokenized text with CLIP
        print(f"Encoding prompt: {prompt}")
        text_tokens = clip.tokenize([prompt], truncate=True).to(DEVICE)
        text_embed = self.diffusion_prior.clip.clip.encode_text(text_tokens)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        text_embed_numpy = text_embed.cpu().detach().numpy().astype("float32")

        # Predict the image embedding from the text embedding
        if image_embed_path.exists() and not overwrite:
            print(f"Loading cached image embed: {image_embed_path}")
            image_embed_numpy = np.load(image_embed_path)
        else:
            print("Inverting CLIP text embedding to image embedding using diffusion prior")
            image_embed = (
                self.diffusion_prior.sample(
                    text=text_tokens,
                    num_samples_per_batch=candidates,
                    cond_scale=cond_scale,
                )
            )
            image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
            image_embed_numpy = image_embed.cpu().detach().numpy().astype("float32")
            np.save(image_embed_path, image_embed_numpy)
            print(f"Saved image embedding to {image_embed_path}")

        return PriorOutput(
            text_embedding=text_embed_numpy[0].tolist(),
            image_embedding=image_embed_numpy[0].tolist(),
        )