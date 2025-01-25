# model_server/main.py

from fastapi import FastAPI
from .core.model_registry import ModelRegistry
from .plugins.llama.model import LlamaModel
from .plugins.gpt2.model import GPT2Model
from .plugins.deepsek.model import DeepSekModel

app = FastAPI()
registry = ModelRegistry()

@app.on_event("startup")
async def startup():
    registry.register("llama", LlamaModel)
    registry.register("gpt2", GPT2Model)
    registry.register("deepsek", DeepSekModel)

@app.post("/v1/chat/completions")
async def chat_completions(messages: list):
    model = await registry.get_model("llama")
    return await model.chat(messages)

@app.post("/v1/completions")
async def completions(prompt: str):
    model = await registry.get_model("llama")
    return await model.generate(prompt)

@app.post("/v1/embeddings")
async def embeddings(text: str):
    model = await registry.get_model("llama")
    return await model.embeddings(text)