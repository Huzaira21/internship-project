"""Caption generation and evaluation utilities."""

from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

_smoothie = SmoothingFunction().method4


def generate_caption(
    image_path, processor, model, device, max_new_tokens=30, num_beams=1
):
    """Generate a caption for a single image."""
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt").to(device)

    if num_beams > 1:
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, num_beams=num_beams
        )
    else:
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)

    return processor.decode(out[0], skip_special_tokens=True)


def compute_bleu(generated, references):
    """Compute BLEU score of a generated caption against one or more reference captions."""
    gen_tokens = word_tokenize(str(generated).lower())
    ref_tokens = [word_tokenize(str(ref).lower()) for ref in references]
    return sentence_bleu(ref_tokens, gen_tokens, smoothing_function=_smoothie)
