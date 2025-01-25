# model_server/plugins/llama/model.py

from typing import Dict, List, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ...core.model_interface import ModelInterface

class LlamaModel(ModelInterface):
   def __init__(self):
       self.model = None
       self.tokenizer = None
       self.device = "cuda" if torch.cuda.is_available() else "cpu"

   async def initialize(self, model_path: str, **kwargs) -> None:
       self.tokenizer = AutoTokenizer.from_pretrained(model_path)
       self.model = AutoModelForCausalLM.from_pretrained(
           model_path,
           device_map="auto",
           torch_dtype=torch.float16
       )

   async def generate(self, prompt: str, **kwargs) -> str:
       inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
       outputs = self.model.generate(**inputs, **kwargs)
       return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

   async def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
       formatted_prompt = ""
       for msg in messages:
           if msg["role"] == "user":
               formatted_prompt += f"User: {msg['content']}\n"
           else:
               formatted_prompt += f"Assistant: {msg['content']}\n"
       
       response = await self.generate(formatted_prompt, **kwargs)
       return {
           "choices": [{
               "message": {
                   "role": "assistant",
                   "content": response
               }
           }]
       }

   async def embeddings(self, text: str) -> List[float]:
       inputs = self.tokenizer(text, return_tensors="pt", padding=True)
       outputs = self.model.get_input_embeddings()(inputs["input_ids"])
       return outputs.mean(dim=1).squeeze().tolist()

   @property
   def model_info(self) -> Dict[str, Any]:
       return {
           "name": "llama",
           "version": "3.1",
           "type": "causal-lm"
       }

   async def unload(self) -> None:
       if self.model:
           del self.model
           self.model = None
       if self.tokenizer:
           del self.tokenizer
           self.tokenizer = None
       torch.cuda.empty_cache()