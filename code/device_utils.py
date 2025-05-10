import torch
import logging

logger = logging.getLogger(__name__)

def get_device():
    """
    Get the best available device with preference order: CUDA > MPS > CPU

    Returns:
        torch.device: The selected device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        logger.info("Using MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    return device

def get_device_map(use_device=None):
    """
    Get the device map for model loading based on available hardware

    Args:
        use_device (torch.device, optional): Force a specific device. Defaults to None.

    Returns:
        str or dict: "auto" for CUDA, specific mapping for MPS/CPU
    """
    device = use_device if use_device is not None else get_device()

    if device.type == "cuda":
        # For CUDA, we can use the "auto" device map for efficient loading
        return "auto"
    elif device.type == "mps":
        # For MPS, we need to be more specific
        return device.type
    else:
        # For CPU
        return device.type

def get_dtype_for_device(device=None, use_bf16=True):
    """
    Get the appropriate dtype for the device

    Args:
        device (torch.device, optional): The device to get dtype for. Defaults to None (uses get_device()).
        use_bf16 (bool, optional): Whether to use bfloat16 when available. Defaults to True.

    Returns:
        torch.dtype: The appropriate dtype for the device
    """
    if device is None:
        device = get_device()

    if device.type == "cuda":
        # For CUDA, we can use bfloat16 if available and requested
        if use_bf16 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16
    elif device.type == "mps":
        # MPS supports float16 but not bfloat16
        return torch.float16
    else:
        # For CPU, use float32
        return torch.float32

def prepare_model_kwargs(model_kwargs, use_quantization=False, device=None):
    """
    Prepare model kwargs based on device and quantization settings

    Args:
        model_kwargs (dict): Original model kwargs
        use_quantization (bool, optional): Whether to use quantization. Defaults to False.
        device (torch.device, optional): Target device. Defaults to None (uses get_device()).

    Returns:
        dict: Updated model kwargs
    """
    if device is None:
        device = get_device()

    # Make a copy to avoid modifying the original
    kwargs = model_kwargs.copy()

    # For MPS, we need special handling
    if device.type == "mps":
        # MPS doesn't support device_map, so we'll load on CPU first then move to MPS
        kwargs["device_map"] = "cpu"
        # Use float16 for MPS
        kwargs["torch_dtype"] = torch.float16
        logger.info("Configured for MPS: loading on CPU first with float16")
    else:
        # Set device map for other devices
        kwargs["device_map"] = get_device_map(device)

        # Set quantization or dtype
        if use_quantization:
            if device.type == "cuda":
                kwargs["load_in_8bit"] = True
            else:
                # Quantization typically only works on CUDA
                logger.warning(f"8-bit quantization not supported on {device.type}, using reduced precision instead")
                kwargs["torch_dtype"] = get_dtype_for_device(device)
        else:
            kwargs["torch_dtype"] = get_dtype_for_device(device)

    return kwargs

def move_model_to_device(model, device=None):
    """
    Move a model to the specified device, with special handling for MPS

    Args:
        model: The model to move
        device (torch.device, optional): Target device. Defaults to None (uses get_device()).

    Returns:
        The model on the target device
    """
    if device is None:
        device = get_device()

    # For MPS, we need to move the model after loading
    if device.type == "mps":
        logger.info("Moving model to MPS device")
        # Check if the model has a device_map attribute
        if hasattr(model, "hf_device_map"):
            logger.info("Model has device_map, moving modules individually")
            # Move each module to MPS
            for module_name in model.hf_device_map:
                if module_name == "":
                    continue  # Skip empty module name
                try:
                    module = model.get_submodule(module_name)
                    module.to(device)
                except Exception as e:
                    logger.warning(f"Error moving module {module_name} to MPS: {e}")
        else:
            # Move the entire model
            try:
                model = model.to(device)
            except Exception as e:
                logger.warning(f"Error moving model to MPS: {e}")
                logger.warning("Continuing with model on CPU")

    return model
