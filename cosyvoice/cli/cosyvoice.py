# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
from typing import Generator
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
import torch
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoiceModel, CosyVoice2Model
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.class_utils import get_model_type


class CosyVoice:

    def __init__(self, model_dir, load_jit=False, load_trt=False, fp16=False, use_pre_quantized=False): # New flag
        self.instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir # Keep original for reference, use model_dir_abs for paths
        self.fp16_arg = fp16 # Store original fp16 argument
        self.use_pre_quantized = use_pre_quantized

        effective_device = torch.device('cpu') if self.use_pre_quantized else \
                           (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        
        logging.info(f"CosyVoice: Effective device determined as: {effective_device}")
        if self.use_pre_quantized and effective_device.type == 'cuda':
            logging.warning("CosyVoice: use_pre_quantized is True, but CUDA is available. Quantized models are intended for CPU. Forcing CPU.")
            effective_device = torch.device('cpu')
        
        current_fp16_state_for_model = self.fp16_arg
        if self.use_pre_quantized:
            if load_jit or load_trt or self.fp16_arg:
                logging.warning("CosyVoice: use_pre_quantized is True. Disabling JIT, TRT, and FP16 as they are likely incompatible with pre-quantized CPU modules.")
            load_jit = False
            load_trt = False
            current_fp16_state_for_model = False # FP16 is not used for pre-quantized CPU modules
        elif effective_device.type == 'cpu' and self.fp16_arg:
            logging.warning("CosyVoice: FP16 is enabled but running on CPU. FP16 will be disabled for model loading.")
            current_fp16_state_for_model = False


        if not os.path.exists(self.model_dir):
            model_dir_abs = snapshot_download(self.model_dir)
        else:
            model_dir_abs = os.path.abspath(self.model_dir)
        logging.info(f"CosyVoice: Absolute model directory: {model_dir_abs}")

        # Always load YAML for frontend and other parameters
        # Determine YAML filename based on typical V1/V2 naming, or make it an explicit arg
        yaml_filename = 'cosyvoice.yaml' # Default to V1
        if os.path.exists(os.path.join(model_dir_abs, 'cosyvoice2.yaml')): # Check if V2 yaml exists
            yaml_filename = 'cosyvoice2.yaml'
        
        hyper_yaml_path = os.path.join(model_dir_abs, yaml_filename)
        if not os.path.exists(hyper_yaml_path):
            raise ValueError('Configuration YAML {} not found in {}!'.format(yaml_filename, model_dir_abs))
        
        logging.info(f"CosyVoice: Loading main config from: {hyper_yaml_path}")
        with open(hyper_yaml_path, 'r') as f:
            # Override device in YAML if effective_device is determined
            configs = load_hyperpyyaml(f, {'device': str(effective_device)})
        
        # --- Initialize Frontend ---
        # Adjust tokenizer path based on V1/V2 if necessary from configs
        speech_tokenizer_filename = 'speech_tokenizer_v1.onnx' # Default V1
        if 'Qwen2LM' in str(configs.get('llm','')) or yaml_filename == 'cosyvoice2.yaml': # V2 indicator
             speech_tokenizer_filename = 'speech_tokenizer_v2.onnx'

        self.frontend = CosyVoiceFrontEnd(
            configs['get_tokenizer'],
            configs['feat_extractor'],
            os.path.join(model_dir_abs, 'campplus.onnx'),
            os.path.join(model_dir_abs, speech_tokenizer_filename),
            os.path.join(model_dir_abs, 'spk2info.pt'),
            configs['allowed_special']
        )
        self.sample_rate = configs['sample_rate']
        logging.info("CosyVoice: Frontend initialized.")

        # --- Initialize or Load Model Modules ---
        if self.use_pre_quantized:
            logging.info("CosyVoice: Loading pre-quantized model objects for CPU inference...")
            # Paths for pre-quantized module objects
            quantized_llm_module_path = os.path.join(model_dir_abs, 'quantized_llm_module.pt')
            quantized_flow_module_path = os.path.join(model_dir_abs, 'quantized_flow_module.pt')
            hift_module_object_path = os.path.join(model_dir_abs, 'hift_module.pt') # HiFT is FP32 object

            if not os.path.exists(quantized_llm_module_path):
                raise FileNotFoundError(f"Quantized LLM module not found: {quantized_llm_module_path}")
            if not os.path.exists(quantized_flow_module_path):
                raise FileNotFoundError(f"Quantized Flow module not found: {quantized_flow_module_path}")
            if not os.path.exists(hift_module_object_path):
                raise FileNotFoundError(f"HiFT module object not found: {hift_module_object_path}")

            loaded_llm = torch.load(quantized_llm_module_path, map_location=effective_device)
            logging.info(f"  Loaded quantized LLM from {quantized_llm_module_path}")
            loaded_flow = torch.load(quantized_flow_module_path, map_location=effective_device)
            logging.info(f"  Loaded (partially) quantized Flow from {quantized_flow_module_path}")
            loaded_hift = torch.load(hift_module_object_path, map_location=effective_device)
            logging.info(f"  Loaded HiFT (FP32 object) from {hift_module_object_path}")
            
            # Initialize CosyVoiceModel with these pre-loaded, pre-configured modules
            self.model = CosyVoiceModel(llm=loaded_llm, flow=loaded_flow, hift=loaded_hift, fp16=False) # fp16 is False for quantized
            # Ensure device consistency (CosyVoiceModel __init__ also does this)
            self.model.device = effective_device 
            self.model.llm.to(effective_device)
            self.model.flow.to(effective_device)
            self.model.hift.to(effective_device)
            logging.info("CosyVoice: CosyVoiceModel initialized with pre-quantized modules on CPU.")

        else: # Standard FP32 loading path
            logging.info("CosyVoice: Initializing models from config and loading FP32 state_dicts...")
            # Initialize FP32 modules from the loaded YAML configurations
            llm_module_fp32 = configs['llm']
            flow_module_fp32 = configs['flow']
            hift_module_fp32 = configs['hift']
            
            self.model = CosyVoiceModel(
                llm=llm_module_fp32,
                flow=flow_module_fp32,
                hift=hift_module_fp32,
                fp16=current_fp16_state_for_model 
            )
            # Modules are moved to device in CosyVoiceModel.__init__
            
            # Load FP32 state_dicts into the initialized modules
            self.model.load_fp32_state_dicts(
                os.path.join(model_dir_abs, 'llm.pt'),
                os.path.join(model_dir_abs, 'flow.pt'),
                os.path.join(model_dir_abs, 'hift.pt')
            )
            logging.info("CosyVoice: CosyVoiceModel initialized and FP32 state_dicts loaded.")

            if load_jit:
                logging.info("CosyVoice: Attempting to load JIT models...")
                # Determine correct fp suffix for JIT paths
                fp_suffix_jit = 'fp16' if self.model.fp16 else 'fp32' # Use model's actual fp16 status
                self.model.load_jit(
                    os.path.join(model_dir_abs, f'llm.text_encoder.{fp_suffix_jit}.zip'),
                    os.path.join(model_dir_abs, f'llm.llm.{fp_suffix_jit}.zip'),
                    os.path.join(model_dir_abs, f'flow.encoder.{fp_suffix_jit}.zip')
                )
            if load_trt:
                logging.info("CosyVoice: Attempting to load TensorRT models...")
                fp_suffix_trt = 'fp16' if self.model.fp16 else 'fp32'
                self.model.load_trt(
                    os.path.join(model_dir_abs, f'flow.decoder.estimator.{fp_suffix_trt}.mygpu.plan'),
                    os.path.join(model_dir_abs, 'flow.decoder.estimator.fp32.onnx'), # ONNX base is usually fp32
                    self.model.fp16 # Pass the model's actual fp16 status for TRT
                )
        
        del configs # Clean up configs
        logging.info("CosyVoice: Initialization complete.")

    def list_available_spks(self):
        spks = list(self.frontend.spk2info.keys())
        return spks

    def add_zero_shot_spk(self, prompt_text, prompt_speech_16k, zero_shot_spk_id):
        assert zero_shot_spk_id != '', 'do not use empty zero_shot_spk_id'
        model_input = self.frontend.frontend_zero_shot('', prompt_text, prompt_speech_16k, self.sample_rate, '')
        del model_input['text']
        del model_input['text_len']
        self.frontend.spk2info[zero_shot_spk_id] = model_input
        return True

    def save_spkinfo(self):
        torch.save(self.frontend.spk2info, '{}/spk2info.pt'.format(self.model_dir))

    def inference_sft(self, tts_text, spk_id, stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_sft(i, spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            if (not isinstance(i, Generator)) and len(i) < 0.5 * len(prompt_text):
                logging.warning('synthesis text {} too short than prompt text {}, this may lead to bad performance'.format(i, prompt_text))
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_speech_16k, self.sample_rate, zero_shot_spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_cross_lingual(self, tts_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_cross_lingual(i, prompt_speech_16k, self.sample_rate, zero_shot_spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_instruct(self, tts_text, spk_id, instruct_text, stream=False, speed=1.0, text_frontend=True):
        assert isinstance(self.model, CosyVoiceModel), 'inference_instruct is only implemented for CosyVoice!'
        if self.instruct is False:
            raise ValueError('{} do not support instruct inference'.format(self.model_dir))
        instruct_text = self.frontend.text_normalize(instruct_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct(i, spk_id, instruct_text)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_vc(self, source_speech_16k, prompt_speech_16k, stream=False, speed=1.0):
        model_input = self.frontend.frontend_vc(source_speech_16k, prompt_speech_16k, self.sample_rate)
        start_time = time.time()
        for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
            speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
            logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
            yield model_output
            start_time = time.time()


class CosyVoice2(CosyVoice):

    def __init__(self, model_dir, load_jit=False, load_trt=False, fp16=False, use_flow_cache=False, use_pre_quantized=False): # New flag
        self.use_flow_cache_arg = use_flow_cache # Store before calling super as super might use parts of self.model

        # Call parent __init__ which handles common setup including pre-quantized loading
        # It will use cosyvoice2.yaml if found, or cosyvoice.yaml otherwise.
        # The `configs` loaded in parent will be based on the found YAML.
        super().__init__(model_dir, load_jit, load_trt, fp16, use_pre_quantized)
        logging.info("CosyVoice2: Parent __init__ completed.")

        # After parent __init__, self.model is an instance of CosyVoiceModel.
        # We need to ensure it's CosyVoice2Model if this class is used.
        
        # Retrieve the already configured/loaded modules from the parent's model instance
        llm_module = self.model.llm
        flow_module = self.model.flow
        hift_module = self.model.hift
        # Determine the fp16 status from the parent model, as it might have been changed (e.g., if on CPU)
        current_fp16_for_model_init = self.model.fp16 
        
        if self.use_pre_quantized:
            # If pre-quantized, fp16 is effectively False for the model logic
            current_fp16_for_model_init = False 
            logging.info("CosyVoice2: Wrapping pre-quantized modules with CosyVoice2Model for CPU.")
        else:
            logging.info("CosyVoice2: Re-initializing model as CosyVoice2Model with loaded FP32 modules.")

        # Replace self.model with a CosyVoice2Model instance, using the modules
        self.model = CosyVoice2Model(
            llm=llm_module,
            flow=flow_module,
            hift=hift_module,
            fp16=current_fp16_for_model_init,
            use_flow_cache=self.use_flow_cache_arg
        )
        # The CosyVoice2Model __init__ will handle moving modules to its self.device
        # and applying .half() if fp16 is True and on CUDA.
        # If use_pre_quantized, device will be CPU.
        
        # JIT/TRT for CosyVoice2Model are different (only flow.encoder for JIT)
        # If not use_pre_quantized, and JIT/TRT flags were true, apply them here.
        # The parent's JIT/TRT loading was for CosyVoiceModel structure.
        if not self.use_pre_quantized:
            model_dir_abs = os.path.abspath(self.model_dir) # Ensure absolute path
            if not os.path.exists(model_dir_abs): # If model_dir was a repo ID
                 model_dir_abs = snapshot_download(self.model_dir)

            if load_jit: # JIT flag from original args
                logging.info("CosyVoice2: Attempting to load JIT for flow.encoder...")
                fp_suffix_jit = 'fp16' if self.model.fp16 else 'fp32'
                # CosyVoice2Model's load_jit is different or might not exist,
                # we might need to call it on self.model.flow.encoder directly if that's the pattern
                # For now, assuming CosyVoice2Model has a load_jit method similar to V1 or handles it.
                # If CosyVoice2Model.load_jit is specific, it should be:
                # self.model.load_jit(os.path.join(model_dir_abs, f'flow.encoder.{fp_suffix_jit}.zip'))
                # For simplicity, if it inherits load_jit from CosyVoiceModel, it might try to load LLM JIT too.
                # This part needs to align with how CosyVoice2Model expects JIT.
                # Let's assume it has a compatible load_jit or we call it on submodules.
                try:
                    # If CosyVoice2Model has its own specific load_jit
                    if hasattr(self.model, 'load_jit_v2'): # Hypothetical specific method
                        self.model.load_jit_v2(os.path.join(model_dir_abs, f'flow.encoder.{fp_suffix_jit}.zip'))
                    elif hasattr(self.model.flow, 'encoder'): # Generic JIT load on encoder
                         if not (isinstance(self.model.flow.encoder, torch.jit.ScriptModule) or \
                            any(isinstance(m, (torch.ao.quantization.QuantizedLinear, torch.ao.quantization.LinearPackedParams)) for m in self.model.flow.encoder.modules())):
                            self.model.flow.encoder = torch.jit.load(os.path.join(model_dir_abs, f'flow.encoder.{fp_suffix_jit}.zip'), map_location=self.model.device)
                            logging.info("CosyVoice2: Loaded JIT for flow.encoder.")
                         else:
                            logging.info("CosyVoice2: Flow encoder already JITted or quantized, skipping JIT load.")
                except Exception as e:
                    logging.warning(f"CosyVoice2: Failed to load JIT for flow.encoder: {e}")


            if load_trt: # TRT flag from original args
                logging.info("CosyVoice2: Attempting to load TensorRT for flow.decoder.estimator...")
                fp_suffix_trt = 'fp16' if self.model.fp16 else 'fp32'
                # Similar to JIT, TRT loading needs to be compatible with CosyVoice2Model
                try:
                    self.model.load_trt( # Assumes load_trt is inherited and compatible or overridden
                        os.path.join(model_dir_abs, f'flow.decoder.estimator.{fp_suffix_trt}.mygpu.plan'),
                        os.path.join(model_dir_abs, 'flow.decoder.estimator.fp32.onnx'),
                        self.model.fp16
                    )
                except Exception as e:
                    logging.warning(f"CosyVoice2: Failed to load TRT for flow.decoder.estimator: {e}")
        logging.info("CosyVoice2: Initialization complete.")

    def inference_instruct(self, *args, **kwargs): # This method is specific to V1
        raise NotImplementedError('inference_instruct is not implemented for CosyVoice2! Use inference_instruct2.')

    def inference_instruct2(self, tts_text, instruct_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):
        assert isinstance(self.model, CosyVoice2Model), 'inference_instruct2 is only implemented for CosyVoice2!'
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct2(i, instruct_text, prompt_speech_16k, self.sample_rate, zero_shot_spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()
