import tempfile

import clip
import numpy as np
import torch
from cog import BaseModel, BasePredictor, Input, Path

from util import load_prior, slugify


class PriorOutput(BaseModel):
    """
    Pydantic class for specifying the return type of the prior prediction API endpoint.
    """

    text_tokens: Path  # we use a `cog.Path` here to return a numpy file instead of e.g. a list of int/float
    text_embedding: Path
    image_embedding: Path


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
    ) -> PriorOutput:
        """
        Load the model into memory to make running multiple predictions efficient

        Args:
            prompt: Text to invert to a CLIP image embed
            cond_scale: How much prior guidance to use.
            candidates: Number of image embeds to draw from in the prior. Increasing may improve performance. 

        Returns:
            A `PriorOutput` object containing the text tokens, text embed and the image embed.
            * `text_tokens` numpy file containing ndarray of type `long`, representing the tokens of the text input.
            * `text_embedding` numpy file containing ndarray of type `float`, included for convenience.
            * `image_embedding` numpy file containing ndarray of type `float`, representing the image embedding predicted by the model.
        """
        # Setup
        assert len(prompt) > 0, "Text input must    be non-empty"
        print(f"Predicting CLIP ViT-L-14 image embed from prompt: {prompt}")

        text_tokens_path = Path(
            self.base_dir, f"ViT-L-14_text_tokens_{slugify(prompt)}.npy"
        )
        text_embed_path = Path(
            self.base_dir, f"ViT-L-14_text_embed_{slugify(prompt)}.npy"
        )
        image_embed_path = Path(
            self.base_dir, f"ViT-L-14_image_embed_{slugify(prompt)}.npy"
        )

        # Tokenize the prompt and save the tokens
        text_tokens = clip.tokenize([prompt], truncate=True).to(DEVICE)
        np.save(
            Path(text_tokens_path), text_tokens.cpu().numpy()
        )  # need this to be a tensor for the next step
        print(f"Saved text tokens to {text_tokens_path}")

        # Predict the image embedding from the text embedding
        print("Inverting CLIP text embedding to image embedding using diffusion prior")
        image_embed = (
            self.diffusion_prior.sample(
                text=text_tokens,
                num_samples_per_batch=candidates,
                cond_scale=cond_scale,
            )
            .cpu()
            .numpy()
        )
        np.save(image_embed_path, image_embed)
        print(f"Saved image embedding to {image_embed_path}")

        # Encode the text embedding as well
        # TODO the sample method above doesnt let you input a text embed, so we compute it twice unnecessarily
        text_embed = (
            self.diffusion_prior.clip.clip.encode_text(text_tokens).cpu().numpy()
        )
        np.save(text_embed_path, text_embed)
        print(f"Saved text embedding to {text_embed_path}")

        return PriorOutput(
            text_tokens=text_tokens_path,
            text_embedding=text_embed_path,
            image_embedding=image_embed_path,
        )