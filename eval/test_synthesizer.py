import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv()

from deepeval.synthesizer import Synthesizer
from deepeval.models.base_model import DeepEvalBaseLLM
from rag.generator import build_llm

class GroqSynthesizerModel(DeepEvalBaseLLM):
    def __init__(self):
        self._model = build_llm(temperature=0.3, max_new_tokens=4096, thinking=False)

    def load_model(self):
        return self._model

    def generate(self, prompt: str) -> str:
        return self._model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        res = await self._model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "ChatGroq"

print("Init model...")
custom_model = GroqSynthesizerModel()
print("Init synthesizer...")
synthesizer = Synthesizer(model=custom_model)
print("Done!")
