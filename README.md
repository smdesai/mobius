![banner.png](banner.png)

# möbius

[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-7289da.svg)](https://discord.gg/WNsvaCtmDe)
[![All Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/FluidInference)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/FluidInference/mobius)
[![Blog](https://img.shields.io/badge/Blog-Read%20More-blue)](https://inference.plus/)

A möbius strip, a surface with only one side and one boundary. What seems like two sides is really one continuous path, reflecting how models in different frameworks remain a single, unified structure as they transform across formats.

Running AI on the edge is difficult and time consuming. We are automating that process with the help of our coding agents and SFT models, `möbius`. We aim to open-source the `möbius` platform in 2026. In the meantime, we will be sharing code, data and models that `möbius` produces from open-sourced models in this repo.

The current focus is converting small models to run on AI Accelerators like ANE and NPUs, and LLMs on edge GPUs. 

We also share some of the conversion process on our [blog](https://inference.plus/).

## Models Folder Structure

The models in this repository are organized using a standardized path structure:

```
models/
│   ├── {class}/
│   │   ├── {name}/
│   │   │   └── {destination}
```

### Path Components

This structure allows for easy navigation and organization of converted models across different frameworks and target platforms.

- **class**: The model class or variant (e.g., `vad`, `llm`, `vllm`, `tts`, `stt`, etc..)
- **name**: Specific model name or version identifier
- **destination**: Target runtime or format (e.g., `coreml`, `onnx`, `openvino`)


## Usage

For usage, look at our other repos like [FluidAudio](https://github.com/FluidInference/FluidAudio) and [fluid-server](https://github.com/FluidInference/fluid-server)

## Citations

```code
@misc{Mobius,
  author = {Fluid Inference},
  title = {möbius},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/FluidInference/mobius}},
  commit = {insert_some_commit_here},
  email = {hello@fluidinference.com}
}
```
