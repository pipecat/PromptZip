import torch
import datasets
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from transformers import AutoTokenizer, AutoModelForCausalLM

class BaseCompressor(ABC):
    def __init__(self, name: str = "BaseCompressor"):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.metrics = {}
        self.compression_time = 0.0
    
    def load_model(self, model_name: str):
        return NotImplementedError

    def compress(self, prompt) -> str:
        raise NotImplementedError
    

class NullCompressor(BaseCompressor):
    def __init__(self, name = "NullCompressor"):
        super().__init__(name)

    def compress(self, prompt) -> str:
        return prompt


class QwenCompressor(BaseCompressor):
    def __init__(self, name = "QwenCompressor"):
        super().__init__(name)

    def load_model(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(torch.device("mps"))
        
    def compress(self, prompt) -> str:
        prompt = prompt + "\n" + "请将以上内容复述一遍：\n\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.7,
            do_sample=True,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)