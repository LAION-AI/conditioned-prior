import re
import unicodedata
from functools import lru_cache

import torch
import torch.nn.functional as F
from dalle2_pytorch import DiffusionPrior, DiffusionPriorNetwork, OpenAIClipAdapter


def l2norm(t):
    return F.normalize(t, dim=-1)


@lru_cache(maxsize=None)
def slugify(value, allow_unicode=False):
    """Taken from https://github.com/django/django/blob/master/django/utils/text.py"""
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def load_prior(model_path, device="cpu", simulated_steps=1000):
    prior_network = DiffusionPriorNetwork(
        dim=768,
        depth=24,
        dim_head=64,
        heads=32,
        normformer=True,
        attn_dropout=5e-2,
        ff_dropout=5e-2,
        num_time_embeds=1,
        num_image_embeds=1,
        num_text_embeds=1,
        num_timesteps=simulated_steps,  # TODO
        ff_mult=4,
    )
    diffusion_prior = DiffusionPrior(
        net=prior_network,
        clip=OpenAIClipAdapter("ViT-L/14"),
        image_embed_dim=768,
        timesteps=1000,
        cond_drop_prob=0.1,
        loss_type="l2",
        condition_on_text_encodings=True,
    )
    diffusion_prior.to(device)
    model_state_dict = torch.load(model_path)
    if "ema_model" in model_state_dict:
        print("Loading EMA Model")
        diffusion_prior.load_state_dict(model_state_dict["ema_model"], strict=True)
    elif "ema_prior_aes_finetune.pth" in model_path:
        diffusion_prior.load_state_dict(model_state_dict, strict=False)
    else:
        print("Loading Standard Model")
        diffusion_prior.load_state_dict(model_state_dict["model"], strict=False)
    del model_state_dict
    return diffusion_prior
