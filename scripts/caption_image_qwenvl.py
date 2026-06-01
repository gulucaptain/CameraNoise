import argparse
from pathlib import Path


DEFAULT_QUESTION = "Please describe the content of the image in detail."


def caption_image_qwenvl(
    model,
    processor,
    reference_image,
    output_dir,
    *,
    question=DEFAULT_QUESTION,
    device="cuda",
    max_new_tokens=128,
):
    """Caption one reference image with a loaded QwenVL model and save caption.txt.

    Parameters
    ----------
    model:
        Loaded QwenVL model. It must support `generate`.
    processor:
        Loaded QwenVL processor. It must support chat template, image encoding, and decoding.
    reference_image:
        Path to one input image.
    output_dir:
        Directory where caption.txt will be written.
    question:
        Prompt sent to QwenVL.
    device:
        Device used for model inputs, for example "cuda", "cuda:0", or "cpu".
    max_new_tokens:
        Maximum number of generated tokens.

    Returns
    -------
    str
        The generated caption text.
    """
    from qwen_vl_utils import process_vision_info

    reference_image = Path(reference_image).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not reference_image.exists():
        raise FileNotFoundError(f"Reference image not found: {reference_image}")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(reference_image),
                },
                {
                    "type": "text",
                    "text": question,
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    caption = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    caption = caption.strip().replace("\n", " ")

    output_dir.mkdir(parents=True, exist_ok=True)
    caption_path = output_dir / "caption.txt"
    caption_path.write_text(caption + "\n", encoding="utf-8")
    return caption


def load_qwenvl(model_path, *, device_map="auto", torch_dtype="auto"):
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def parse_args():
    parser = argparse.ArgumentParser(description="Caption one image with QwenVL and save caption.txt.")
    parser.add_argument("--model-path", required=True, type=str, help="QwenVL model path.")
    parser.add_argument("--reference-image", required=True, type=Path, help="Input reference image path.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to save caption.txt.")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Caption prompt.")
    parser.add_argument("--device", default="cuda", help="Input tensor device, for example cuda, cuda:0, or cpu.")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map for model loading.")
    parser.add_argument("--torch-dtype", default="auto", help="Transformers torch_dtype for model loading.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum generation length.")
    return parser.parse_args()


def main():
    args = parse_args()
    model, processor = load_qwenvl(
        args.model_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )
    caption = caption_image_qwenvl(
        model,
        processor,
        args.reference_image,
        args.output_dir,
        question=args.question,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    caption = caption.replace("\n", " ")
    print(caption)
    print(f"caption saved: {Path(args.output_dir).expanduser().resolve() / 'caption.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
