# Conditioned Prior (WIP)

<a href="https://replicate.com/laion-ai/conditioned-prior" target="_blank"><img src="https://img.shields.io/static/v1?label=Replicate&message=Demo and API&color=pink"></a>

Weights and code by [@nousr](https://twitter.com/nousr_)


Predict a CLIP image embedding from its text embedding using a diffusion prior.

This code is part of an effort to replicate the models laid out in [Hierarchical Text-Conditional Image Generation with CLIP Latents](https://arxiv.org/abs/2204.06125).

## Requirements

* nvidia GPU
* [docker](https://docs.docker.com/get-docker/)
* [cog](https://github.com/replicate/cog/)

## Quick start

```sh
cog predict r8.im/laion-ai/conditioned-prior \
    -i prompt="..." \
    -i candidates=2 \
    -i cond_scale=1.0
```

### Parameters

* prompt: Text to invert to a CLIP image embed

* cond_scale: How much prior guidance to use.

* candidates: Number of image embeds to draw from in the prior. Increasing may improve performance. 


### Output

A `PriorOutput` - typed dictionary containing the text tokens, text embed and the image embed.

* `PriorOutput.text_tokens` - ".npy" file containing ndarray of type `long`, representing the tokens of the text input. Included for convenience.

* `PriorOutput.text_embedding` - ".npy" file containing ndarray of type `float`, included for convenience.

* `PriorOutput.image_embedding` ".npy" file containing ndarray of type `float`, representing the image embedding predicted by the model.

## Intended use

Anytime you need a CLIP image embed but only have a text description. For instance:

* Use as input to models that accept CLIP embeds such as CLIP-guided VQGAN, diffusion to improve generations.

* Use to improve performance on lookup tasks

## Special Thanks

* [LAION](https://discord.gg/uPMftTmrvS) for support, resources, and community

* [Stability AI](https://stability.ai/) for compute which makes these models possible

* [lucidrains](https://github.com/lucidrains) for spearheading the open-source replication of DALLE 2

## Caveats and recommendations

Just to avoid any confusion, this research is a recreation of (one part of) OpenAI's DALLE2 paper. It is _not_, "DALLE2", the product/service from OpenAI you may have seen on the web.

## Contribute

* Install [docker](https://docs.docker.com/get-docker/).
* Install [cog](https://github.com/replicate/cog/).

```sh
git clone https://github.com/laion-ai/conditoned-prior.git && cd conditioned-prior
```

### Build the docker image from scratch

Download the "slim" weights:

```sh
wget https://huggingface.co/laion/DALLE2-PyTorch/resolve/main/prior_ema_fp16.pth
```

Then, run:

```sh
cog build -t "my-custom-conditioned-prior"
```

### Local prediction flask endpoint

```sh
docker run -d -p 5000:5000 --gpus=all 'my-custom-conditioned-prior'
```

A `POST` route `/predictions` will now trigger the model to be run. Weights are only loaded into GPU memory once upon running `docker run`, making repeated API calls faster.

```sh
curl http://localhost:5000/predictions -X POST -H "Content-Type: application/json" \
  -d '{"input": {
    "prompt": "...",
    "candidates": "2",
    "cond_scale": "1.0"
  }}'
```

### Push a fork to your own Replicate account

First, edit the `image` property in [cog.yaml](/cog.yaml)

```yaml
# ...
image: "" # TODO put your own url here after creating a model on Replicate.
build:
  gpu: true
  python_version: "3.8"
# ...
```

Make sure you are logged in:

```sh
cog login
```

and push your docker image to Replicate:

```sh
cog push
```

### Update the official laion-ai Replicate demo/api
If you need to change the Replicate demo uploaded to `replicate.com/laion-ai/conditioned-prior`, you will need to be invited to be part of the laion-ai org on Replicate. Reach out to @afiaka87, @robvanvolt, @christophschuhmann, or @rom1504 if you need to.
