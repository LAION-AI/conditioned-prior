# Conditioned Prior (WIP)

Weights and code by [@nousr](https://twitter.com/nousr_)

Predict a CLIP image embedding from its text embedding using a diffusion prior.

This code is part of an effort to replicate the models laid out in [Hierarchical Text-Conditional Image Generation with CLIP Latents](https://arxiv.org/abs/2204.06125).

## Special Thanks

* [LAION](https://discord.gg/uPMftTmrvS) for support, resources, and community

* [Stability AI](https://stability.ai/) for compute which makes these models possible

* [lucidrains](https://github.com/lucidrains) for spearheading the open-source replication of DALLE 2

## Quick start (docker required)

* Install [docker](https://docs.docker.com/get-docker/)
* Install [cog](https://github.com/replicate/cog/)

```sh
cog predict r8.im/laion-ai/conditioned-prior \
    -i prompt="..." \
    -i candidates=2 \
    -i cond_scale=1.0
```

## Intended use

Anytime you need a CLIP image embed but only have a text description. For instance:

* Use as input to models that accept CLIP embeds such as CLIP-guided VQGAN, diffusion to improve generations.

* Use to improve performance on lookup tasks

## Caveats and recommendations

Just to avoid any confusion, this research is a recreation of (one part of) OpenAI's DALLE2 paper. It is _not_, "DALLE2", the product/service from OpenAI you may have seen on the web.
