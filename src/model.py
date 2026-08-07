"""Model loading utilities for the BLIP image captioning model."""

from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

DEFAULT_MODEL_NAME = "Salesforce/blip-image-captioning-base"


def load_blip_model(model_name=DEFAULT_MODEL_NAME):
    """Load the BLIP processor and model, and move the model to the best available device."""
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return processor, model, device


def freeze_vision_encoder(model):
    """Freeze the vision encoder parameters so only the text decoder is trainable."""
    for param in model.vision_model.parameters():
        param.requires_grad = False
    return model
