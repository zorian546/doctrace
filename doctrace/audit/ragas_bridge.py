"""
Adapter that lets RAGAS use the local Qwen2.5-3B-Instruct as its judge model.

Implements RAGAS's BaseRagasLLM so RAGAS runs offline and repeatably, with no
third-party API involved.
"""

from langchain_core.outputs import Generation, LLMResult
from ragas.llms.base import BaseRagasLLM


class LocalRagasJudge(BaseRagasLLM):
    """BaseRagasLLM implementation backed by the local Qwen2.5-3B-Instruct."""

    def __init__(self):
        super().__init__()
        from doctrace.answering.local_model import complete as _complete
        self._complete = _complete

    def _run(self, prompt) -> LLMResult:
        reply = self._complete(None, prompt.to_string(), max_new_tokens=1024)
        return LLMResult(generations=[[Generation(text=reply)]])

    def generate_text(self, prompt, n: int = 1, temperature: float = 1e-8,
                      stop=None, callbacks=None) -> LLMResult:
        return self._run(prompt)

    async def agenerate_text(self, prompt, n: int = 1, temperature=None,
                             stop=None, callbacks=None) -> LLMResult:
        return self._run(prompt)
