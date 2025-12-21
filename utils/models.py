
# A collection of APIs to use for generating model responses
# OpenAI/Anthropic models are supported via their respective APIs below
# Other models are routed to LocalAPI, which assumes a VLLM instance running on localhost:8000
# Additional APIs (e.g., Google, Together, etc.) may need to be added
# Only asyncronous apis are supported
# Non-asnyc requests can be handled, but chat() should still be async, so include something like await asyncio.sleep(0)

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import os

import backoff
import openai
import anthropic
import asyncio
import re


class ChatAPI(ABC):

    @abstractmethod
    def __init__(self, model):
        pass

    @abstractmethod
    async def chat(self, messages):
        pass


class OpenAIAPI(ChatAPI):

    def __init__(self, model: str):
        self.model = model
        self.client = openai.AsyncClient(
            api_key=os.environ.get("OPENAI_API_KEY"))

    @backoff.on_exception(backoff.fibo, (openai.OpenAIError), max_tries=5, max_value=30)
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        
        if self.model.startswith("o1"):
            if messages[0]["role"] == "system":
                system_message = messages.pop(0)["content"]
                user_message = messages[0]["content"]
                messages[0] = {
                    "role": "user",
                    "content": f"<|BEGIN_SYSTEM_MESSAGE|>\n{system_message.strip()}\n<|END_SYSTEM_MESSAGE|>\n\n{user_message}"
                }

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
        elif self.model.startswith("gpt-5"):
            response = await self.client.responses.create(
                        model=self.model,
                        input=messages,
                        reasoning={ "effort": "low" },
                        text={ "verbosity": "low" },
                    )
            return response.output_text
        else:   
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **kwargs,
                )
            except openai.BadRequestError as e:
                msg = str(e)
                # 当提示上下文长度不足时，自动降低 max_tokens 再重试
                if ("maximum context length is" in msg) and ("max_tokens" in msg or "max_completion_tokens" in msg):
                    m_ctx = re.search(r"maximum context length is (\d+)", msg)
                    m_in = re.search(r"your request has (\d+) input tokens", msg)
                    if m_ctx and m_in:
                        ctx_limit = int(m_ctx.group(1))
                        input_tokens = int(m_in.group(1))
                        allowed = max(1, ctx_limit - input_tokens - 1)
                        current = kwargs.get("max_tokens")
                        if current is None or allowed < int(current):
                            retry_kwargs = dict(kwargs)
                            retry_kwargs["max_tokens"] = allowed
                            response = await self.client.chat.completions.create(
                                model=self.model,
                                messages=messages,
                                **retry_kwargs,
                            )
                        else:
                            raise
                else:
                    raise
        return response.choices[0].message.content


class AnthropicAPI(ChatAPI):

    def __init__(self, model: str):
        self.model = model
        self.client = anthropic.AsyncClient(
            api_key=os.environ.get("ANTHROPIC_API_KEY"))

    @backoff.on_exception(backoff.fibo, anthropic.AnthropicError, max_tries=5, max_value=30)
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        
        if messages[0]["role"] == "system":
            system_message = messages[0]["content"]
            messages = messages[1:]
        else:
            system_message = ""

        response = await self.client.messages.create(
            model=self.model,
            messages=messages,
            system=system_message,
            # max_tokens=8192, # 4096 for claude-3-*
            **kwargs,
        )
        
        return response.content[0].text


class GeminiAPI(OpenAIAPI):
    """Google Gemini API.
    
    Models include gemini-1.5-pro-001 etc.
    """
    def __init__(self, model: str):
        import google.auth
        import google.auth.transport.requests

        # Programmatically get an access token, need to setup your google cloud account properly,
        # and get `gcloud auth application-default login` to be run first
        creds, _project = google.auth.default()
        auth_req = google.auth.transport.requests.Request()
        creds.refresh(auth_req)
        # Note: the credential lives for 1 hour by default (https://cloud.google.com/docs/authentication/token-types#at-lifetime); after expiration, it must be refreshed.

        project_id = creds.quota_project_id
        # Pass the Vertex endpoint and authentication to the OpenAI SDK
        self.model = f"google/{model}"
        self.client = openai.AsyncClient(
            base_url=f"https://us-central1-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/us-central1/endpoints/openapi",
            api_key=creds.token,
        )

    @backoff.on_exception(backoff.fibo, (openai.OpenAIError), max_tries=10, max_value=30)
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        message = response.choices[0].message
        if message is None:
            # This happens for Google Gemini under high concurrency
            raise openai.OpenAIError("No response from Google Gemini")
        return message.content


class TogetherAPI(ChatAPI):

    def __init__(self, model: str):
        self.model = model
        self.client = openai.AsyncClient(
            api_key=os.environ.get("TOGETHER_API_KEY"),
            base_url="https://api.together.xyz/v1"
        )

    @backoff.on_exception(backoff.fibo, (openai.OpenAIError), max_tries=5, max_value=30)
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs,
            )
        except openai.BadRequestError as e:
            msg = str(e)
            if ("maximum context length is" in msg) and ("max_tokens" in msg or "max_completion_tokens" in msg):
                m_ctx = re.search(r"maximum context length is (\d+)", msg)
                m_in = re.search(r"your request has (\d+) input tokens", msg)
                if m_ctx and m_in:
                    ctx_limit = int(m_ctx.group(1))
                    input_tokens = int(m_in.group(1))
                    allowed = max(1, ctx_limit - input_tokens - 1)
                    current = kwargs.get("max_tokens")
                    if current is None or allowed < int(current):
                        retry_kwargs = dict(kwargs)
                        retry_kwargs["max_tokens"] = allowed
                        response = await self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            **retry_kwargs,
                        )
                    else:
                        raise
            else:
                raise
        return response.choices[0].message.content

def get_chat_api_from_model(model: str) -> ChatAPI:
    if model.startswith("gpt") or model.startswith("o1"):
        return OpenAIAPI(model)
    if model.startswith("claude"):
        return AnthropicAPI(model)
    if model.startswith("gemini"):
        return GeminiAPI(model)
    if model.startswith("meta-llama"):
        return TogetherAPI(model)
    raise ValueError(
        f"Unsupported remote model: {model}. "
        "Use model_type='local' and models.get_local_api(...) for local deployments."
    )


# =====================
# Local backends support
# =====================

class VLLMAPI(ChatAPI):

    def __init__(self, model: str, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model
        base_url = base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        api_key = api_key or os.environ.get("VLLM_API_KEY", "EMPTY")
        self.client = openai.AsyncClient(base_url=base_url, api_key=api_key)

    @backoff.on_exception(backoff.fibo, (openai.OpenAIError), max_tries=5, max_value=30)
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        if "seed" not in kwargs:
            import os
            kwargs["seed"] = int(os.environ.get("SEED", 42))
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs,
            )
        except openai.BadRequestError as e:
            msg = str(e)
            # 上下文超限：自动降低 max_tokens 并重试
            if ("maximum context length is" in msg) and ("max_tokens" in msg or "max_completion_tokens" in msg):
                m_ctx = re.search(r"maximum context length is (\d+)", msg)
                m_in = re.search(r"your request has (\d+) input tokens", msg)
                if m_ctx and m_in:
                    ctx_limit = int(m_ctx.group(1))
                    input_tokens = int(m_in.group(1))
                    allowed = max(1, ctx_limit - input_tokens - 1)
                    current = kwargs.get("max_tokens")
                    if current is None or allowed < int(current):
                        retry_kwargs = dict(kwargs)
                        retry_kwargs["max_tokens"] = allowed
                        response = await self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            **retry_kwargs,
                        )
                    else:
                        raise
            else:
                raise
        return response.choices[0].message.content


class HFTransformersAPI(ChatAPI):
    """Local HuggingFace Transformers backend.

    Loads a causal LM via transformers and implements a simple chat-style interface
    by concatenating messages into a single prompt.
    """

    def __init__(self, model: str):
        self.model_name = model
        # Lazy imports to avoid importing transformers if not needed
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        dtype_str = os.environ.get("HF_DTYPE", "auto")
        if dtype_str == "auto":
            torch_dtype = None
        else:
            torch_dtype = getattr(torch, dtype_str, None)

        device_map = os.environ.get("HF_DEVICE_MAP", "auto")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )

    @staticmethod
    def _format_messages(messages: List[Dict[str, str]]) -> str:
        # Simple prompt formatting; adapt as needed for better chat templates
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)

    @backoff.on_exception(backoff.fibo, Exception, max_tries=3, max_value=10)
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # Run sync generation in a thread to keep async interface
        import torch

        prompt = self._format_messages(messages)
        temperature = kwargs.get("temperature", 0.0)
        top_p = kwargs.get("top_p", 1.0)
        max_tokens = kwargs.get("max_tokens", 1024)

        def _generate() -> str:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    do_sample=(temperature and temperature > 0.0),
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Return only assistant continuation after the last 'assistant:' tag if present
            if "assistant:" in text:
                text = text.split("assistant:")[-1].strip()
            return text

        return await asyncio.to_thread(_generate)


def get_local_api(model: str, backend: Optional[str] = None) -> ChatAPI:
    """Return a local chat API implementation.

    backend: "vllm" (default) uses an OpenAI-compatible endpoint at VLLM_BASE_URL.
             "hf"   uses local transformers for direct inference.
             Can be controlled by env LOCAL_BACKEND.
    """
    backend = (backend or os.environ.get("LOCAL_BACKEND", "vllm")).lower()
    if backend == "hf":
        return HFTransformersAPI(model)
    # default vLLM (OpenAI-compatible server)
    return VLLMAPI(model)
