#%%
import io
import re
import typing
import unicodedata
import urllib
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from dalle2_pytorch import DiffusionPrior, DiffusionPriorNetwork, OpenAIClipAdapter
from einops import rearrange, repeat

def l2norm(t):
    return F.normalize(t, dim=-1)


@torch.inference_mode()
def sample_prior(
    diffusion_prior,
    text_embed,
    text_encodings,
    text_mask,
    num_samples_per_batch=2,
    cond_scale=1.0,
):
    batch_size = text_embed.shape[0]
    image_embed_dim = diffusion_prior.image_embed_dim
    text_cond = dict(
        text_embed=text_embed
    )  # sample 2 image embeddings, choose the top 1 similarity, as judged by CLIP
    # if self.condition_on_text_encodings:
    text_cond = {**text_cond, "text_encodings": text_encodings, "mask": text_mask}
    image_embeds = diffusion_prior.p_sample_loop(
        (batch_size, image_embed_dim), text_cond=text_cond, cond_scale=cond_scale
    )
    image_embeds /= diffusion_prior.image_embed_scale
    text_embeds = text_cond["text_embed"]
    text_embeds = rearrange(text_embeds, "(b r) d -> b r d", r=num_samples_per_batch)
    image_embeds = rearrange(image_embeds, "(b r) d -> b r d", r=num_samples_per_batch)
    text_image_sims = torch.einsum(
        "b r d, b r d -> b r", l2norm(text_embeds), l2norm(image_embeds)
    )
    top_sim_indices = text_image_sims.topk(k=1).indices
    top_sim_indices = repeat(top_sim_indices, "b 1 -> b 1 d", d=image_embed_dim)
    top_image_embeds = image_embeds.gather(1, top_sim_indices)
    return rearrange(top_image_embeds, "b 1 d -> b d")


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


def download_image(url):
    urllib_request = urllib.request.Request(
        url,
        data=None,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"
        },
    )
    with urllib.request.urlopen(urllib_request, timeout=10) as r:
        img_stream = io.BytesIO(r.read())
    return img_stream


def load_prior(model_path):
    """
    Loads the prior model and returns it. Doesn't move it to the gpu.
    **Note** - this is a modified version of the original function to allow for the use of slim fp16 checkpoints.
    """
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
        num_timesteps=1000,
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
    state_dict = torch.load(model_path, map_location="cpu")
    diffusion_prior.load_state_dict(state_dict, strict=True)
    diffusion_prior.eval()
    return diffusion_prior


@torch.no_grad()
def get_text_emb(openai_clip_model, text_tokens):
    text_emb = openai_clip_model.encode_text(text_tokens)
    text_emb /= text_emb.norm(dim=-1, keepdim=True)
    text_emb = text_emb.cpu().detach().numpy().astype("float32")
    return text_emb



def similarity(image_embedding, text_embedding):
    image_embedding = image_embedding / np.linalg.norm(image_embedding)
    text_embedding = text_embedding / np.linalg.norm(text_embedding)
    return np.inner(image_embedding, text_embedding)


def rerank_and_sample(image_embeddings, text_embedding, samples=None, strategy="top"):
    """
    Here we take the prompt, generate n number of embeddings and rerank them by cosine similarity to the text embedding,
    then take a linspace of N and to see the variation in the performance of the prior
    """
    if samples is None:
        samples = len(image_embeddings)
    reranked = sorted(
        list(image_embeddings), key=lambda img_emb: similarity(img_emb, text_embedding)
    )
    if strategy == "top":
        sampled_embeddings = np.array(reranked[-samples:])
    elif strategy == "even":
        sample_indices = np.linspace(0, len(reranked) - 1, num=samples, dtype=int)
        sampled_embeddings = np.array([reranked[i] for i in sample_indices])
    rankings = [similarity(emb, text_embedding) for emb in sampled_embeddings]
    print(rankings, rankings[0], rankings[-1])
    # return sampled_embeddings
    return sampled_embeddings, rankings