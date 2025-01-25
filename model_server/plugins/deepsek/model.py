# model_server/plugins/deepsek/model.py

from typing import Dict, List, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ...core.model_interface import ModelInterface

class DeepSekModel(ModelInterface):
   def __init__(self):
       self.model = None
       self.tokenizer = None
       self.device = "cuda" if torch.cuda.is_available() else "cpu"
       self.config = None

   async def initialize(self, model_path: str, **kwargs):
       self.tokenizer = AutoTokenizer.from_pretrained(model_path)
       self.model = AutoModelForCausalLM.from_pretrained(
           model_path,
           device_map="auto",
           torch_dtype=torch.float16,
           **kwargs
       )
       self.model.eval()
       
   async def generate(self, prompt: str, **kwargs):
       inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
       with torch.no_grad():
           outputs = self.model.generate(
               **inputs,
               max_new_tokens=kwargs.get('max_length', 512),
               temperature=kwargs.get('temperature', 0.7),
               top_p=kwargs.get('top_p', 0.95),
               top_k=kwargs.get('top_k', 50),
               repetition_penalty=kwargs.get('repetition_penalty', 1.1)
           )
       return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

   async def chat(self, messages: List[Dict[str, str]], **kwargs):
       chat_template = "<|system|>You are a helpful AI assistant.</s><|user|>{user_msg}</s><|assistant|>"
       formatted_messages = []
       
       for msg in messages:
           if msg['role'] == 'user':
               formatted_messages.append(chat_template.format(user_msg=msg['content']))
           elif msg['role'] == 'assistant':
               formatted_messages.append(f"{msg['content']}</s>")
               
       prompt = "".join(formatted_messages)
       response = await self.generate(prompt, **kwargs)
       
       return {
           "choices": [{
               "message": {
                   "role": "assistant",
                   "content": response
               }
           }]
       }

   async def embeddings(self, text: str):
       inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
       with torch.no_grad():
           outputs = self.model.get_input_embeddings()(inputs["input_ids"].to(self.device))
       pooled_output = outputs.mean(dim=1)
       return pooled_output.cpu().numpy().tolist()[0]

   @property
   def model_info(self):
       return {
           "name": "deepsek",
           "version": "33b",
           "architecture": "decoder-only",
           "context_length": 4096,
           "parameters": "33B"
       }

   async def unload(self):
       if self.model:
           del self.model
           self.model = None
       if self.tokenizer:
           del self.tokenizer
           self.tokenizer = None
       torch.cuda.empty_cache()