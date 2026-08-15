# -*- coding: utf-8 -*-
"""Qwen 模型封装模块"""

import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

QWEN_MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'Qwen1.5-4B-Chat')


class QwenWrapper:
    def __init__(
        self,
        model_path=None,
        device=None,
        torch_dtype=None,
        device_map=None,
        load_in_4bit=False,
        load_in_8bit=False,
        **kwargs
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.model_path = model_path or QWEN_MODEL_DIR
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        
        if device_map is not None:
            self.device = device or 'cuda'
        else:
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.torch_dtype = torch_dtype
        if self.torch_dtype is None and torch.cuda.is_available():
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = self.torch_dtype or torch.float32
        self.device_map = device_map
        
        load_kwargs: dict = {
            'trust_remote_code': True,
            'low_cpu_mem_usage': True,
        }
        
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            load_kwargs['quantization_config'] = quantization_config
            load_kwargs['device_map'] = 'auto'
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs['quantization_config'] = quantization_config
            load_kwargs['device_map'] = 'auto'
        else:
            load_kwargs['torch_dtype'] = self.torch_dtype
            load_kwargs['device_map'] = 'auto'
            if device_map is not None:
                load_kwargs['device_map'] = device_map
        
        load_kwargs.update(kwargs)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True,
        )
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)
            self.model.eval()
            if load_in_4bit or load_in_8bit:
                self.device = next(self.model.parameters()).device
            else:
                self.device = torch.device(self.device)
        except Exception as e:
            raise

    def chat_single_turn(self, prompt, system="You are a helpful assistant.", **gen_kwargs):
        """单轮对话"""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        gen_kwargs.setdefault('max_new_tokens', 512)
        gen_kwargs.setdefault('eos_token_id', self.tokenizer.eos_token_id)
        
        if gen_kwargs.get('do_sample') is False:
            gen_kwargs.pop('temperature', None)
            gen_kwargs.pop('top_p', None)
        elif gen_kwargs.get('temperature', 1.0) < 0.01:
            gen_kwargs['do_sample'] = False
            gen_kwargs.pop('temperature', None)
            gen_kwargs.pop('top_p', None)
        else:
            gen_kwargs['do_sample'] = True
            gen_kwargs.setdefault('temperature', 0.7)
            gen_kwargs.setdefault('top_p', 0.9)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                **gen_kwargs
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
