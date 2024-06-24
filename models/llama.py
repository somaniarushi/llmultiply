import os
import time
from typing import Any

import requests
from models.base import BaseModel

MAX_ALLOWED_TRIALS = 10

URL = "https://api.together.xyz/v1/chat/completions"

HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": f"Bearer {os.environ.get('TOGETHER_BEARER_TOKEN')}",
}


class LlamaSampler(BaseModel):
    """
    Sample from Together's llama3 chat completion API
    """

    def __init__(
        self,
        model: str = "meta-llama/Llama-3-8b-chat-hf",
    ):
        self.model = model

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def generate(self, prompt: str, max_tokens_to_generate: int, temperature: float, top_k: int) -> str:
        trial = 0
        while trial < MAX_ALLOWED_TRIALS:
            try:
                message_list = [
                    self._pack_message("user", prompt),
                ]
                payload = {
                    "model": self.model,
                    "messages": message_list,
                    "max_tokens": max_tokens_to_generate,
                    "temperature": temperature,
                    "top_k": top_k,
                    "stop": ["|<im_end>|", "assistant"],
                }
                response = requests.post(URL, json=payload, headers=HEADERS)
                assert (
                    response.status_code == 200
                ), f"Rate limit detected: {response.text}"
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
        raise Exception("Rate limit exception too many times, cannot continue further")