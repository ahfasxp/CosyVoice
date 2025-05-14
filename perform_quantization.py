import torch
import torch.quantization
from hyperpyyaml import load_hyperpyyaml
import os
import shutil
import sys

# Add path to CosyVoice if this script is not in the project root
# Adjust this path if necessary
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.')) 
sys.path.append(project_root)
# If your Matcha-TTS is in a different location, adjust this too
sys.path.append(os.path.join(project_root, 'third_party/Matcha-TTS'))


# Import relevant CosyVoice modules
# Ensure PYTHONPATH is correct or sys.path.append above covers the cosyvoice directory
try:
    from cosyvoice.cli.model import CosyVoiceModel # Base class, not directly instantiated here for quantization
    # Import specific module classes defined in your YAML
    from cosyvoice.llm.llm import TransformerLM 
    from cosyvoice.transformer.encoder import ConformerEncoder, TransformerEncoder
    from cosyvoice.flow.flow import MaskedDiffWithXvec 
    from cosyvoice.flow.decoder import ConditionalDecoder
    from cosyvoice.hifigan.generator import HiFTGenerator
except ImportError as e:
    print(f"Error importing CosyVoice modules: {e}")
    print("Please ensure PYTHONPATH is set correctly or sys.path.append points to the correct directories.")
    sys.exit(1)


def quantize_and_save_submodules(config_path, fp32_model_dir, output_quantized_dir):
    print(f"Starting quantization process...")
    print(f"Config path: {config_path}")
    print(f"FP32 model dir: {fp32_model_dir}")
    print(f"Output quantized dir: {output_quantized_dir}")

    if not os.path.exists(output_quantized_dir):
        os.makedirs(output_quantized_dir)
        print(f"Created output directory: {output_quantized_dir}")

    device = torch.device('cpu') # Quantization is performed on CPU

    # 1. Load configuration
    print("Loading YAML configuration...")
    # Ensure device in config is also CPU if it's used for module initialization from YAML
    with open(config_path, 'r') as f:
        configs = load_hyperpyyaml(f, लड़ाई={'device': 'cpu'}) 
    print("YAML configuration loaded.")

    # Initialize FP32 modules from configuration
    # This is crucial to have the correct model architecture before loading FP32 state_dicts
    print("Initializing FP32 modules from configuration...")
    # These will call the __init__ of the respective nn.Module classes from your YAML
    llm_fp32 = configs['llm']
    flow_fp32 = configs['flow']
    hift_fp32 = configs['hift']

    llm_fp32.to(device).eval()
    flow_fp32.to(device).eval()
    hift_fp32.to(device).eval()
    print("FP32 modules initialized and set to eval mode on CPU.")

    # 2. Load FP32 state_dicts into the initialized modules
    print("Loading FP32 state_dicts...")
    llm_fp32_path = os.path.join(fp32_model_dir, 'llm.pt')
    flow_fp32_path = os.path.join(fp32_model_dir, 'flow.pt')
    hift_fp32_path = os.path.join(fp32_model_dir, 'hift.pt')

    llm_fp32.load_state_dict(torch.load(llm_fp32_path, map_location=device))
    print(f"Loaded LLM FP32 state_dict from {llm_fp32_path}")
    
    flow_fp32.load_state_dict(torch.load(flow_fp32_path, map_location=device))
    print(f"Loaded Flow FP32 state_dict from {flow_fp32_path}")
    
    # For hift, state_dict keys might have 'generator.' prefix
    hift_state_dict_raw = torch.load(hift_fp32_path, map_location=device)
    if any('generator.' in k for k in hift_state_dict_raw.keys()):
        hift_state_dict = {k.replace('generator.', ''): v for k, v in hift_state_dict_raw.items()}
    else:
        hift_state_dict = hift_state_dict_raw
    hift_fp32.load_state_dict(hift_state_dict)
    print(f"Loaded HiFT FP32 state_dict from {hift_fp32_path}")
    
    # --- Quantize LLM ---
    print("\nQuantizing LLM module...")
    llm_to_quantize = llm_fp32 # This is the TransformerLM instance
    llm_to_quantize.eval() 

    # Quantize sub-modules within TransformerLM (text_encoder and llm internal)
    # This is specific to the TransformerLM structure defined in your YAML
    if hasattr(llm_to_quantize, 'text_encoder') and isinstance(llm_to_quantize.text_encoder, torch.nn.Module):
        print("  Quantizing LLM's text_encoder...")
        llm_to_quantize.text_encoder.eval()
        quantized_text_encoder = torch.quantization.quantize_dynamic(
            llm_to_quantize.text_encoder, {torch.nn.Linear}, dtype=torch.qint8
        )
        llm_to_quantize.text_encoder = quantized_text_encoder # Replace with quantized version
        print("  LLM's text_encoder quantized.")
    
    if hasattr(llm_to_quantize, 'llm') and isinstance(llm_to_quantize.llm, torch.nn.Module): # This is the internal llm (e.g., TransformerEncoder)
        print("  Quantizing LLM's internal llm module...")
        llm_to_quantize.llm.eval()
        quantized_internal_llm = torch.quantization.quantize_dynamic(
            llm_to_quantize.llm, {torch.nn.Linear}, dtype=torch.qint8
        )
        llm_to_quantize.llm = quantized_internal_llm # Replace with quantized version
        print("  LLM's internal llm module quantized.")
    
    # Save the entire LLM module object, which now contains quantized parts
    quantized_llm_module_path = os.path.join(output_quantized_dir, 'quantized_llm_module.pt')
    torch.save(llm_to_quantize, quantized_llm_module_path)
    print(f"Quantized LLM module (object) saved to {quantized_llm_module_path}")

    # --- Quantize Flow (Encoder and Decoder Estimator) ---
    print("\nQuantizing Flow module parts...")
    flow_to_modify = flow_fp32 # This is the MaskedDiffWithXvec instance
    flow_to_modify.eval()

    if hasattr(flow_to_modify, 'encoder') and isinstance(flow_to_modify.encoder, torch.nn.Module):
        print("  Quantizing Flow's encoder...")
        flow_to_modify.encoder.eval()
        quantized_flow_encoder = torch.quantization.quantize_dynamic(
            flow_to_modify.encoder, {torch.nn.Linear}, dtype=torch.qint8
        )
        flow_to_modify.encoder = quantized_flow_encoder # Replace in the flow_to_modify object
        print("  Flow's encoder quantized.")
    
    if hasattr(flow_to_modify, 'decoder') and \
       hasattr(flow_to_modify.decoder, 'estimator') and \
       isinstance(flow_to_modify.decoder.estimator, torch.nn.Module):
        print("  Quantizing Flow's decoder estimator...")
        flow_to_modify.decoder.estimator.eval()
        quantized_flow_decoder_estimator = torch.quantization.quantize_dynamic(
            flow_to_modify.decoder.estimator, {torch.nn.Linear}, dtype=torch.qint8
        )
        flow_to_modify.decoder.estimator = quantized_flow_decoder_estimator # Replace
        print("  Flow's decoder estimator quantized.")
    
    # Save the entire Flow module object, which now contains quantized parts
    # Other parts (e.g., length_regulator) remain FP32 with their loaded weights.
    quantized_flow_module_path = os.path.join(output_quantized_dir, 'quantized_flow_module.pt')
    torch.save(flow_to_modify, quantized_flow_module_path)
    print(f"Modified Flow module (object with quantized parts) saved to {quantized_flow_module_path}")
    
    # --- HiFT (Not dynamically quantized, save FP32 module object) ---
    print("\nSaving HiFT module (as FP32 object)...")
    # hift_fp32 already has its state_dict loaded
    hift_module_object_path = os.path.join(output_quantized_dir, 'hift_module.pt')
    torch.save(hift_fp32, hift_module_object_path)
    print(f"HiFT module (FP32 object) saved to {hift_module_object_path}")

    # --- Copy configuration and other supporting files ---
    print("\nCopying supporting files...")
    shutil.copy2(config_path, os.path.join(output_quantized_dir, os.path.basename(config_path)))
    
    support_files_to_copy = []
    # Heuristic to determine if it's CosyVoice v1 or v2 based on typical filenames in config
    if "Qwen2LM" in str(configs.get('llm', '')) or "speech_tokenizer_v2" in str(configs.get('frontend',{}).get('speech_tokenizer_path','')): # Check for V2 indicators
        yaml_filename = 'cosyvoice2.yaml'
        speech_tokenizer_filename = 'speech_tokenizer_v2.onnx'
        if os.path.exists(os.path.join(fp32_model_dir, 'CosyVoice-BlankEN')):
             shutil.copytree(os.path.join(fp32_model_dir, 'CosyVoice-BlankEN'), os.path.join(output_quantized_dir, 'CosyVoice-BlankEN'), dirs_exist_ok=True)
    else: # Assume CosyVoice v1
        yaml_filename = 'cosyvoice.yaml'
        speech_tokenizer_filename = 'speech_tokenizer_v1.onnx'
    
    support_files_to_copy = ['campplus.onnx', speech_tokenizer_filename, 'spk2info.pt', yaml_filename]


    for fname in support_files_to_copy:
        source_fpath = os.path.join(fp32_model_dir, fname)
        target_fpath = os.path.join(output_quantized_dir, fname) # Ensure target has same name
        if os.path.exists(source_fpath):
            shutil.copy2(source_fpath, target_fpath)
            print(f"  Copied {fname}")
        else:
            # If the primary YAML (cosyvoice.yaml or cosyvoice2.yaml) is not found in source, it's an issue.
            # The config_path variable should point to the correct one.
            if fname == os.path.basename(config_path): 
                 print(f"  Copied {os.path.basename(config_path)} (from config_path argument)")
            else:
                 print(f"  Warning: {fname} not found at {source_fpath}, not copied.")

    print("Supporting files copied.")
    print("\nQuantization process complete.")
    print(f"Quantized model objects and assets saved to: {output_quantized_dir}")


if __name__ == '__main__':
    # --- CONFIGURATION ---
    # Replace with appropriate paths for your model
    # This is your original fine-tuned FP32 model directory
    FP32_MODEL_DIR = 'pretrained_models/ft-model' 
    # This is the YAML configuration file for your FP32 model
    # Ensure this filename matches what's in FP32_MODEL_DIR (cosyvoice.yaml or cosyvoice2.yaml)
    CONFIG_FILENAME = 'cosyvoice.yaml' # Change to 'cosyvoice2.yaml' if that's your model type
    CONFIG_FILE = os.path.join(FP32_MODEL_DIR, CONFIG_FILENAME) 
    # This is the directory where the quantized model objects will be saved
    QUANTIZED_OUTPUT_DIR = 'pretrained_models/quantized_dynamic_ft-model' 
    # --- END CONFIGURATION ---

    if not os.path.exists(FP32_MODEL_DIR):
        print(f"Error: FP32 model directory not found: {FP32_MODEL_DIR}")
        sys.exit(1)
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: Configuration YAML not found: {CONFIG_FILE}")
        print(f"Please ensure {CONFIG_FILENAME} exists in {FP32_MODEL_DIR} or update CONFIG_FILENAME.")
        sys.exit(1)

    quantize_and_save_submodules(CONFIG_FILE, FP32_MODEL_DIR, QUANTIZED_OUTPUT_DIR)
```

**Changes to `cosyvoice/cli/model.py`:**

```python
cosyvoice/cli/model.py
<<<<<<< SEARCH
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool = False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        if self.fp16 is True:
            self.llm.half()
            self.flow.half()
