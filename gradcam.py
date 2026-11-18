"""Grad-CAM utilities for the Kidney CT classifier."""
import matplotlib
import numpy as np
import tensorflow as tf

LAST_CONV = "top_activation"  # last convolutional activation in EfficientNetB0


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=LAST_CONV, pred_index=None):
    """Return a Grad-CAM heatmap (values 0-1) for one image batch of shape (1, H, W, 3)."""
    grad_model = tf.keras.Model(
        model.input,
        [model.get_layer(last_conv_layer_name).output, model.outputs[0]],
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(img, heatmap, alpha=0.4):
    """Overlay a heatmap on an image. img: float array 0-255, shape (H, W, 3)."""
    heatmap = np.uint8(255 * heatmap)
    jet = matplotlib.colormaps["jet"](np.arange(256))[:, :3]
    jet_heat = jet[heatmap]
    jet_heat = tf.image.resize(jet_heat, (img.shape[0], img.shape[1])).numpy() * 255
    return np.uint8(np.clip(jet_heat * alpha + img, 0, 255))
