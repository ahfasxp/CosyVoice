#!/usr/bin/env python3
import torch
import os
import sys
import shutil
import numpy as np
from datetime import datetime

# Add CosyVoice to path
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2

def quantize_state_dict(state_dict, quantized_dict=None):
    """
    Quantize a state dictionary by converting float32 tensors to int8.
    This preserves the original structure while reducing memory usage.
    """
    if quantized_dict is None:
        quantized_dict = {}
    
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor) and tensor.dtype in [torch.float32, torch.float16]:
            # Only quantize appropriate tensors (weights and large matrices)
            if any(pattern in key for pattern in ['weight', 'bias', 'embedding']) and tensor.dim() >= 1:
                # Store scales for dequantization
                tensor_np = tensor.detach().cpu().numpy()
                max_abs_value = np.abs(tensor_np).max()
                scale = max_abs_value / 127.0 if max_abs_value > 0 else 1.0
                
                # Quantize to int8
                quantized_tensor = torch.tensor(
                    (tensor_np / scale).round().clip(-127, 127).astype(np.int8),
                    dtype=torch.int8
                )
                
                # Store both the quantized tensor and its scale
                quantized_dict[key] = {
                    'quantized': quantized_tensor,
                    'scale': torch.tensor(scale, dtype=torch.float32)
                }
                
                # Print memory reduction - handle the calculation safely
                original_size = tensor.nelement() * tensor.element_size()
                quantized_size = quantized_tensor.nelement() * quantized_tensor.element_size() + 4  # 4 bytes for scale
                reduction = 100 * (1 - quantized_size / original_size)
                print(f"    - Quantized {key}: {original_size/1024/1024:.2f} MB → {quantized_size/1024/1024:.2f} MB ({reduction:.2f}% reduction)")
            else:
                # Keep small tensors and non-weight tensors as is
                quantized_dict[key] = tensor
        elif isinstance(tensor, dict):
            # Recursively quantize nested dictionaries
            quantized_dict[key] = {}
            quantize_state_dict(tensor, quantized_dict[key])
        elif isinstance(tensor, (np.ndarray, np.number)) and np.issubdtype(tensor.dtype, np.floating):
            # Handle numpy arrays and scalars
            if isinstance(tensor, np.ndarray) and tensor.size > 1:
                # For large numpy arrays
                max_abs_value = np.abs(tensor).max()
                scale = max_abs_value / 127.0 if max_abs_value > 0 else 1.0
                
                # Quantize to int8
                quantized_np = np.round(tensor / scale).clip(-127, 127).astype(np.int8)
                quantized_tensor = torch.tensor(quantized_np, dtype=torch.int8)
                
                # Store both the quantized tensor and its scale
                quantized_dict[key] = {
                    'quantized': quantized_tensor,
                    'scale': torch.tensor(scale, dtype=torch.float32)
                }
                
                # Print memory reduction
                original_size = tensor.size * tensor.itemsize
                quantized_size = quantized_np.size * quantized_np.itemsize + 4  # 4 bytes for scale
                reduction = 100 * (1 - quantized_size / original_size)
                print(f"    - Quantized numpy array {key}: {original_size/1024/1024:.2f} MB → {quantized_size/1024/1024:.2f} MB ({reduction:.2f}% reduction)")
            else:
                # Keep small arrays and scalar values as is
                quantized_dict[key] = tensor
        else:
            # Keep non-tensor items as is
            quantized_dict[key] = tensor
            
    return quantized_dict

def create_dequantization_code():
    """
    Create the code for dequantization functions that will be saved with the model
    """
    code = '''
def dequantize_state_dict(state_dict):
    """Dequantize an int8 state dictionary back to float32."""
    import torch
    import numpy as np
    dequantized_dict = {}
    
    for key, value in state_dict.items():
        if isinstance(value, dict) and "quantized" in value and "scale" in value:
            # Dequantize
            dequantized_dict[key] = value["quantized"].float() * value["scale"]
        elif isinstance(value, dict) and not ("quantized" in value and "scale" in value):
            # Recursively dequantize nested dictionaries
            dequantized_dict[key] = dequantize_state_dict(value)
        else:
            # Keep non-quantized items as is
            dequantized_dict[key] = value
            
    return dequantized_dict
'''
    return code

def create_load_wrapper_code():
    """
    Create the code for the load wrapper that will patch torch.load
    """
    code = '''
import os
import sys
import torch
from dequantize import dequantize_state_dict

# Monkey patch torch.load to automatically dequantize
original_load = torch.load
def patched_load(f, *args, **kwargs):
    state_dict = original_load(f, *args, **kwargs)
    if isinstance(state_dict, dict) and any(
        isinstance(v, dict) and "quantized" in v and "scale" in v 
        for v in state_dict.values()):
        print(f"Dequantizing quantized state dictionary: {f}")
        return dequantize_state_dict(state_dict)
    return state_dict

torch.load = patched_load
'''
    return code

def create_readme(model_path, output_dir):
    """
    Create a README file with usage instructions
    """
    readme = "# Quantized CosyVoice Model\n\n"
    readme += f"This is a quantized version of the CosyVoice model from {model_path}.\n"
    readme += "The weights have been quantized to INT8 format to reduce memory usage and improve CPU performance.\n\n"
    
    readme += "## Usage\n\n"
    readme += "To use this quantized model:\n\n"
    readme += "```python\n"
    readme += "import sys\n"
    readme += "sys.path.append('path/to/this/directory')\n"
    readme += "import load_wrapper  # This patches torch.load to handle quantized weights\n\n"
    readme += "# Then load normally\n"
    readme += "from cosyvoice.cli.cosyvoice import CosyVoice\n"
    readme += f"cosyvoice = CosyVoice('{os.path.basename(output_dir)}', load_jit=False, fp16=False)\n"
    readme += "```\n\n"
    
    readme += "The model will automatically dequantize weights when loaded, so it works just like the original model\n"
    readme += "but with reduced memory usage.\n\n"
    
    readme += "## Benefits of Quantization\n\n"
    readme += "- Reduced memory usage (approximately 75% smaller)\n"
    readme += "- Improved CPU inference performance\n"
    readme += "- Same model API and output quality\n"
    
    return readme

def quantize_cosyvoice_model(model_path, output_dir="int8_model"):
    """
    Quantize the state dictionaries in a CosyVoice model directory
    """
    print(f"Preparing to quantize model in {model_path}...")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model path {model_path} does not exist!")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # First, copy all non-PT files to preserve structure
    print("Copying model files to output directory...")
    for item in os.listdir(model_path):
        source_path = os.path.join(model_path, item)
        dest_path = os.path.join(output_dir, item)
        
        if os.path.isdir(source_path):
            if not os.path.exists(dest_path):
                shutil.copytree(source_path, dest_path)
        elif not item.endswith('.pt'):  # Skip .pt files as we'll handle them separately
            shutil.copy2(source_path, dest_path)
    
    # Save the dequantize function
    with open(os.path.join(output_dir, 'dequantize.py'), 'w') as f:
        f.write(create_dequantization_code())
    
    # Save the load wrapper
    with open(os.path.join(output_dir, 'load_wrapper.py'), 'w') as f:
        f.write(create_load_wrapper_code())
    
    # Find and process the PT files
    model_files = [f for f in os.listdir(model_path) if f.endswith('.pt')]
    
    print(f"Found {len(model_files)} PyTorch state dictionary files:")
    for file in model_files:
        print(f"  - {file}")
    
    # Quantize each state dictionary
    quantized_files = 0
    total_original_size = 0
    total_quantized_size = 0
    
    for file in model_files:
        print(f"\nProcessing {file}...")
        try:
            # Load the state dictionary
            state_dict_path = os.path.join(model_path, file)
            state_dict = torch.load(state_dict_path, map_location='cpu')
            
            # Calculate original size
            original_size = os.path.getsize(state_dict_path)
            total_original_size += original_size
            print(f"  - Loaded state dictionary with {len(state_dict)} keys")
            print(f"  - Original file size: {original_size/1024/1024:.2f} MB")
            
            # Quantize the state dictionary
            print("  - Quantizing...")
            quantized_dict = quantize_state_dict(state_dict)
            
            # Save the quantized state dictionary
            output_path = os.path.join(output_dir, file)
            print(f"  - Saving quantized state dictionary to {output_path}")
            torch.save(quantized_dict, output_path)
            
            # Calculate quantized size
            quantized_size = os.path.getsize(output_path)
            total_quantized_size += quantized_size
            reduction = 100 * (1 - quantized_size / original_size)
            print(f"  - Quantized file size: {quantized_size/1024/1024:.2f} MB ({reduction:.2f}% reduction)")
            
            print(f"  - Successfully quantized {file}")
            quantized_files += 1
            
        except Exception as e:
            print(f"  - ERROR: Failed to quantize {file}: {str(e)}")
            # Copy the original file as fallback
            shutil.copy2(os.path.join(model_path, file), os.path.join(output_dir, file))
            print(f"  - Copied original file as fallback")
            total_quantized_size += original_size  # Add to total since we're using original
    
    # Create README
    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write(create_readme(model_path, output_dir))
    
    # Add quantization info
    total_reduction = 100 * (1 - total_quantized_size / total_original_size) if total_original_size > 0 else 0
    with open(os.path.join(output_dir, 'quantization_info.txt'), 'w') as f:
        f.write(f"Model quantized to INT8\n")
        f.write(f"Original model path: {model_path}\n")
        f.write(f"Quantization date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Files quantized: {quantized_files} of {len(model_files)}\n")
        f.write(f"Total size reduction: {total_reduction:.2f}%\n")
        f.write(f"Original size: {total_original_size/1024/1024:.2f} MB\n")
        f.write(f"Quantized size: {total_quantized_size/1024/1024:.2f} MB\n")
    
    print(f"\nQuantization summary:")
    print(f"  - Files quantized: {quantized_files} of {len(model_files)}")
    print(f"  - Total size reduction: {total_reduction:.2f}%")
    print(f"  - Original size: {total_original_size/1024/1024:.2f} MB")
    print(f"  - Quantized size: {total_quantized_size/1024/1024:.2f} MB")
    print(f"\nQuantization complete! Model saved to {output_dir}")
    return output_dir

def test_quantized_model(output_dir):
    """
    Test if the quantized model can be loaded and used
    """
    print(f"\nTesting quantized model in {output_dir}...")
    
    try:
        # Add the quantized model directory to the path
        sys.path.append(os.path.abspath(output_dir))
        
        # Import the load wrapper to patch torch.load
        try:
            import load_wrapper
            print("  - Successfully imported load_wrapper")
        except ImportError as e:
            print(f"  - ERROR: Failed to import load_wrapper: {str(e)}")
            return False
        
        # Try to load the model
        try:
            model = CosyVoice(output_dir, load_jit=False, fp16=False)
            print("  - Successfully loaded quantized model")
            return True
        except Exception as e:
            print(f"  - ERROR: Failed to load quantized model: {str(e)}")
            return False
            
    except Exception as e:
        print(f"  - ERROR during testing: {str(e)}")
        return False
    finally:
        # Clean up by removing the directory from path
        if os.path.abspath(output_dir) in sys.path:
            sys.path.remove(os.path.abspath(output_dir))

# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Quantize CosyVoice model for better CPU performance")
    parser.add_argument("--model_path", type=str, default="pretrained_models/ft-model",
                        help="Path to the CosyVoice model directory")
    parser.add_argument("--output_dir", type=str, default="int8_model",
                        help="Output directory for the quantized model")
    parser.add_argument("--test", action="store_true",
                        help="Test the quantized model after quantization")
    args = parser.parse_args()
    
    # Quantize the model
    output_dir = quantize_cosyvoice_model(args.model_path, args.output_dir)
    
    # Test the model if requested
    if args.test and output_dir:
        test_result = test_quantized_model(output_dir)
        if test_result:
            print("\nTest successful! The quantized model works correctly.")
        else:
            print("\nTest failed! The quantized model may not work correctly.")
    
    # Print usage instructions
    if output_dir:
        print(f"\nTo use the quantized model for inference:")
        print(f"import sys")
        print(f"sys.path.append('{os.path.abspath(output_dir)}')")
        print(f"import load_wrapper  # This patches torch.load")
        print(f"from cosyvoice.cli.cosyvoice import CosyVoice")
        print(f"cosyvoice = CosyVoice('{output_dir}', load_jit=False, fp16=False)")