#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for device management (CPU, GPU, MPS)
"""

import os
import logging
import torch

logger = logging.getLogger(__name__)

def get_device():
    """Get the appropriate device for training/inference"""
    if torch.backends.mps.is_available():
        logger.info("Using MPS (Metal Performance Shaders)")
        return "mps"
    elif torch.cuda.is_available():
        logger.info("Using CUDA")
        return "cuda"
    else:
        logger.info("Using CPU")
        return "cpu"

def prepare_model_kwargs(model_name, device):
    """Prepare kwargs for model loading based on device"""
    kwargs = {}
    
    if device == "mps":
        # For MPS, load on CPU first with float16
        logger.info("Configured for MPS: loading on CPU first with float16")
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = "auto"
    elif device == "cuda":
        # For CUDA, use float16 and device_map
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = "auto"
    else:
        # For CPU, use device_map only
        kwargs["device_map"] = "auto"
    
    # Special handling for Qwen model
    if model_name == "qwen" and os.environ.get("DISABLE_SLIDING_WINDOW_ATTENTION", "false").lower() == "true":
        logger.info("Disabling Sliding Window Attention for Qwen model")
        kwargs["sliding_window"] = None
    
    return kwargs

def move_model_to_device(model, device):
    """Move model to the specified device"""
    logger.info(f"Moving model to {device} device")
    
    if hasattr(model, "device_map") and model.device_map is not None:
        logger.info("Model has device_map, moving modules individually")
        # Model already has device_map, no need to move
        return model
    
    return model.to(device)
