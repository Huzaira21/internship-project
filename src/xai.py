"""Explainability utilities: Grad-CAM and occlusion-based importance for BLIP."""

import numpy as np
import cv2
from PIL import Image

_activations = {}
_gradients = {}


def register_gradcam_hooks(model):
    """Register forward/backward hooks on the last layer of the vision encoder."""

    def save_activation(name):
        def hook(module, input, output):
            _activations[name] = output

        return hook

    def save_gradient(name):
        def hook(module, grad_input, grad_output):
            _gradients[name] = grad_output[0]

        return hook

    target_layer = model.vision_model.encoder.layers[-1]
    target_layer.register_forward_hook(save_activation("vision_last_layer"))
    target_layer.register_full_backward_hook(save_gradient("vision_last_layer"))


def generate_gradcam(image_path, processor, model, device):
    """Generate a Grad-CAM heatmap for a single image."""
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt").to(device)

    outputs = model.generate(**inputs, max_new_tokens=20)
    generated_caption = processor.decode(outputs[0], skip_special_tokens=True)

    model.zero_grad()
    pixel_values = inputs["pixel_values"]
    pixel_values.requires_grad_()

    vision_outputs = model.vision_model(pixel_values=pixel_values)
    image_embeds = vision_outputs.last_hidden_state
    target = image_embeds.mean()
    target.backward()

    act = _activations["vision_last_layer"][0].detach().cpu().numpy()
    grad = _gradients["vision_last_layer"][0].detach().cpu().numpy()

    weights = np.mean(grad, axis=0)
    cam = np.dot(act, weights)
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)

    num_patches = int(np.sqrt(cam.shape[0]))
    cam_map = cam[-(num_patches**2) :].reshape(num_patches, num_patches)
    cam_resized = cv2.resize(cam_map, (raw_image.width, raw_image.height))

    return raw_image, cam_resized, generated_caption


def overlay_heatmap(raw_image, cam_resized, alpha=0.5):
    """Overlay a Grad-CAM heatmap on top of the original image."""
    img_np = np.array(raw_image)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)


def occlusion_importance(image_path, processor, model, device, patch_size=32):
    """Compute a simple occlusion-based importance map for an image."""
    raw_image = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.array(raw_image)

    inputs = processor(raw_image, return_tensors="pt").to(device)
    original_out = model.generate(**inputs, max_new_tokens=20)
    original_caption = processor.decode(original_out[0], skip_special_tokens=True)

    h, w, _ = img_array.shape
    importance_map = np.zeros((h // patch_size, w // patch_size))

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            occluded = img_array.copy()
            occluded[i : i + patch_size, j : j + patch_size] = 128

            occ_inputs = processor(Image.fromarray(occluded), return_tensors="pt").to(
                device
            )
            occ_out = model.generate(**occ_inputs, max_new_tokens=20)
            occ_caption = processor.decode(occ_out[0], skip_special_tokens=True)

            diff = len(set(original_caption.split()) - set(occ_caption.split()))
            importance_map[i // patch_size, j // patch_size] = diff

    return raw_image, importance_map, original_caption
