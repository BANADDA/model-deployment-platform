# model_server/plugins/gpt2/model.py

from typing import Dict, List, Any
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from ...core.model_interface import ModelInterface

class GPT2Model(ModelInterface):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    async def initialize(self, model_path: str, **kwargs):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)

    async def generate(self, prompt: str, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, **kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    async def chat(self, messages: List[Dict[str, str]], **kwargs):
        combined_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        response = await self.generate(combined_prompt, **kwargs)
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response
                }
            }]
        }

    async def embeddings(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model.transformer(inputs["input_ids"].to(self.device))
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().tolist()

    @property
    def model_info(self):
        return {
            "name": "gpt2",
            "version": "1.0",
            "type": "causal-lm"
        }

    async def unload(self):
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()