from __future__ import annotations

import os
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import typer

from transformers import AutoModelForTokenClassification, AutoTokenizer
import wtpsplit.models  # registers SubwordXLM config/model types

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_MODEL_ID = "segment-any-text/sat-3l-sm"


from conversion_utils import (
    Conversion,
    apply_conversion,
    update_manifest_model_name,
)

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def parse_conversion_type(value: str | None) -> Conversion:
    if value is None:
        return Conversion.NONE
    value_str = value.strip()
    if not value_str:
        return Conversion.NONE

    try:
        return Conversion[value_str.upper()]
    except KeyError as exc:
        raise typer.BadParameter(
            f"Invalid conversion type '{value}'. "
            "Choose from 'none', 'prune', 'quantize', or 'palettize'."
        ) from exc


def parse_conversion_types(
    values: tuple[str, ...] | list[str] | None,
) -> list[Conversion]:
    if not values:
        return [Conversion.NONE]

    parsed: list[Conversion] = []
    for item in values:
        parsed.append(parse_conversion_type(item))
    return parsed


@app.command()
def convert(
    model_id: str = typer.Option(
        DEFAULT_MODEL_ID,
        "--model-id",
        help="Model identifier to download from HuggingFace's model hub",
    ),
    output_dir: Path = typer.Option(
        Path("sat_coreml"),
        help="Directory where mlpackages and metadata will be written",
    ),
    conversion_types: list[str] = typer.Option(
        None,
        "--conversion-type",
        "-c",
        help=(
            "Conversion methods to apply to the model. "
            "Repeat the option to chain conversions "
            "(allowed: none, prune, quantize, palettize; default: none)."
        ),
    ),
):

    conversions_to_apply = parse_conversion_types(conversion_types)

    model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        return_dict=False,
        torchscript=True,
        trust_remote_code=True,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained("facebookAI/xlm-roberta-base")
    tokenized = tokenizer(
        ["Sample input text to trace the model."],
        return_tensors="pt",
        max_length=512,  # token sequence length
        padding="max_length",
    )

    traced_model = torch.jit.trace(
        model,
        (tokenized["input_ids"], tokenized["attention_mask"])
    )

    outputs = [ct.TensorType(name="output")]

    mlpackage = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                f"{name}",
                shape=tensor.shape,
                dtype=np.int32,
            )
            for name, tensor in tokenized.items()
        ],
        outputs=outputs,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS18,
    )

    try:
        new_model = mlpackage
        for conversion in conversions_to_apply:
            new_model = apply_conversion(new_model, conversion)
    except ValueError as e:
        print(e)
        return

    saved_name = "SaT"
    saved_path = output_dir / f"{saved_name}.mlpackage"
    new_model.save(saved_path)

    manifest_file = saved_path / "Manifest.json"
    update_manifest_model_name(manifest_file, saved_name)

if __name__ == "__main__":
    app()
