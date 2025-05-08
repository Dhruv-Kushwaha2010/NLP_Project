import torch
import logging
from device_utils import get_device, get_device_map, get_dtype_for_device, prepare_model_kwargs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_device_detection():
    """Test device detection and preference order"""
    device = get_device()
    logger.info(f"Selected device: {device}")
    
    # Check device capabilities
    if device.type == "cuda":
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA device capability: {torch.cuda.get_device_capability(0)}")
        logger.info(f"CUDA device properties: {torch.cuda.get_device_properties(0)}")
    elif device.type == "mps":
        logger.info("Using Apple Metal Performance Shaders (MPS)")
    else:
        logger.info("Using CPU")
    
    return device

def test_device_map(device):
    """Test device map generation"""
    device_map = get_device_map(device)
    logger.info(f"Device map for {device}: {device_map}")
    return device_map

def test_dtype_selection(device):
    """Test dtype selection based on device"""
    dtype = get_dtype_for_device(device)
    logger.info(f"Selected dtype for {device}: {dtype}")
    
    # Test with different settings
    dtype_no_bf16 = get_dtype_for_device(device, use_bf16=False)
    logger.info(f"Selected dtype for {device} (no bf16): {dtype_no_bf16}")
    
    return dtype

def test_model_kwargs(device):
    """Test model kwargs preparation"""
    # Test with base kwargs
    base_kwargs = {"trust_remote_code": True}
    
    # Test without quantization
    kwargs_no_quant = prepare_model_kwargs(base_kwargs, use_quantization=False, device=device)
    logger.info(f"Model kwargs without quantization: {kwargs_no_quant}")
    
    # Test with quantization
    kwargs_with_quant = prepare_model_kwargs(base_kwargs, use_quantization=True, device=device)
    logger.info(f"Model kwargs with quantization: {kwargs_with_quant}")
    
    return kwargs_no_quant, kwargs_with_quant

if __name__ == "__main__":
    logger.info("Testing device utilities...")
    
    # Run tests
    device = test_device_detection()
    test_device_map(device)
    test_dtype_selection(device)
    test_model_kwargs(device)
    
    logger.info("All tests completed!")
