
from langchain_aws import BedrockLLM
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from decouple import config
import json
import requests # Mantenha se usar em outro lugar, mas httpx é usado aqui
from typing import Optional, List, Any, Iterator, AsyncIterator, Dict
import httpx # Usar httpx para chamadas sync e async
import asyncio

# Importações LangChain necessárias
from langchain.llms.base import LLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from langchain.schema.output import GenerationChunk # Importante para streaming

class ChatOllama(LLM):
    """
    Custom LangChain LLM wrapper for Ollama's generate API.

    Handles both synchronous (_call) and asynchronous streaming (_astream).
    """
    model: str = "llama3.2"  # Modelo padrão
    api_url: str = config("OLLAMA_API_URL", default="http://ollama:11434/api/generate")
    temperature: float = 0.0
    request_timeout: Optional[float] = None

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "ollama"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "api_url": self.api_url,
            "temperature": self.temperature,
            "request_timeout": self.request_timeout,
        }

    def _prepare_payload(self, prompt: str, stream: bool, **kwargs) -> Dict[str, Any]:
        """Prepara o payload JSON para a API Ollama."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": stream,
            # Adiciona quaisquer outros kwargs relevantes para a API Ollama
            **kwargs
        }
        # Remove chaves None ou não relevantes para a API
        payload = {k: v for k, v in payload.items() if v is not None}
        return payload

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Método síncrono NÃO-streaming. Envia o prompt e retorna a resposta completa.
        """
        # Adiciona 'stop' aos kwargs se fornecido e não None
        if stop:
            kwargs["stop"] = stop

        payload = self._prepare_payload(prompt, stream=False, **kwargs)

        try:
            with httpx.Client(timeout=self.request_timeout) as client:
                response = client.post(self.api_url, json=payload)
                response.raise_for_status() # Levanta erro para status HTTP >= 400
                data = response.json()
                full_response = data.get("response", "").strip()

                # Callback opcional com a resposta final (LangChain não tem um padrão claro para _call final aqui)
                # Mas podemos simular a chegada de um único "token" grande
                if run_manager:
                    run_manager.on_llm_new_token(full_response)

                return full_response

        except httpx.RequestError as e:
            raise ValueError(f"Erro ao conectar com a API Ollama em {self.api_url}: {e}") from e
        except httpx.HTTPStatusError as e:
            raise ValueError(f"Erro na API Ollama ({e.response.status_code}): {e.response.text}") from e
        except Exception as e:
            raise RuntimeError(f"Erro inesperado ao processar resposta Ollama: {e}") from e

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """
        Método de streaming SÍNCRONO.
        """
        if stop:
            kwargs["stop"] = stop

        payload = self._prepare_payload(prompt, stream=True, **kwargs)

        try:
            with httpx.Client(timeout=self.request_timeout) as client:
                with client.stream("POST", self.api_url, json=payload) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            try:
                                token_info = json.loads(line)
                                token = token_info.get("response", "")
                                chunk = GenerationChunk(text=token) # Cria o chunk esperado pelo LangChain
                                yield chunk
                                # Chama o callback para cada novo token
                                if run_manager:
                                    run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                                # Verifica se é o fim do stream (Ollama envia 'done: true' no final)
                                if token_info.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                print(f"Aviso: Falha ao decodificar linha JSON do stream Ollama: {line}")
                                continue # Pula linhas malformadas
        except httpx.RequestError as e:
            if run_manager: run_manager.on_llm_error(e)
            raise ValueError(f"Erro ao conectar com a API Ollama em {self.api_url}: {e}") from e
        except httpx.HTTPStatusError as e:
            if run_manager: run_manager.on_llm_error(e)
            raise ValueError(f"Erro na API Ollama ({e.response.status_code}): {e.response.text}") from e
        except Exception as e:
            if run_manager: run_manager.on_llm_error(e)
            raise RuntimeError(f"Erro inesperado ao processar stream Ollama: {e}") from e

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        """
        Método de streaming ASSÍNCRONO.
        É este que será usado por `chain.astream`.
        """
        if stop:
            kwargs["stop"] = stop

        payload = self._prepare_payload(prompt, stream=True, **kwargs)

        try:
            async with httpx.AsyncClient(timeout=self.request_timeout) as client:
                async with client.stream("POST", self.api_url, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                token_info = json.loads(line)
                                token = token_info.get("response", "")
                                # Cria o chunk esperado pelo LangChain
                                chunk = GenerationChunk(text=token)
                                yield chunk
                                # Chama o callback assíncrono para cada novo token
                                if run_manager:
                                    await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                                # Verifica se é o fim do stream
                                if token_info.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                print(f"Aviso: Falha ao decodificar linha JSON do stream Ollama: {line}")
                                continue # Pula linhas malformadas
        except httpx.RequestError as e:
            if run_manager: await run_manager.on_llm_error(e)
            raise ValueError(f"Erro ao conectar com a API Ollama em {self.api_url}: {e}") from e
        except httpx.HTTPStatusError as e:
            if run_manager: await run_manager.on_llm_error(e)
            raise ValueError(f"Erro na API Ollama ({e.response.status_code}): {e.response.text}") from e
        except Exception as e:
            if run_manager: await run_manager.on_llm_error(e)
            raise RuntimeError(f"Erro inesperado ao processar stream Ollama: {e}") from e


# --- Função de Inicialização Atualizada ---

# Add available models here (mantido como no original)
bedrock_model_ids = [
    "mistral.mixtral-8x7b-instruct-v0:1",
    "amazon.titan-text-premier-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "cohere.command-r-plus-v1:0",
]

def str_to_bool(value):
    if isinstance(value, bool): # Já trata booleano diretamente
        return value
    if value is None:
        return False
    return str(value).lower() in ['true', '1', 't', 'y', 'yes']

def initialize_llm(model_id=None, openai_api_key=None):
    openai_api_key = openai_api_key or config("OPENAI_API_KEY", default=None) # Evitar erro se não definida
    use_azure = config("USE_AZURE", default=False, cast=str_to_bool)
    # Corrigido para ler USE_OLLAMA corretamente
    use_ollama = config("USE_OLLAMA", default=True, cast=str_to_bool) # Default True se não definido

    print(f"--- Inicializando LLM ---")
    print(f"OpenAI Key fornecida: {'Sim' if openai_api_key else 'Não'}")
    print(f"Usar Azure: {use_azure}")
    print(f"Usar Ollama: {use_ollama}")

    llm = None # Inicializa llm como None

    if model_id in bedrock_model_ids:
        print(f"Modelo Bedrock selecionado: {model_id}")
        llm = BedrockLLM(model_id=model_id)
    elif use_azure:
        print("Configurando para Azure OpenAI...")
        # Validações básicas das configs Azure
        azure_endpoint = config("AZURE_OPENAI_ENDPOINT", default="")
        azure_deployment = config("AZURE_DEPLOYMENT", default="")
        api_version = config("AZURE_API_VERSION", default="")
        if not all([azure_endpoint, azure_deployment, api_version, openai_api_key]):
             print("AVISO: Faltam configurações para Azure OpenAI (Endpoint, Deployment, API Version, API Key).")
             # Decide se quer lançar erro ou tentar outra opção
             # Por agora, apenas avisa e tentará outra opção se houver
        else:
            llm = AzureChatOpenAI(
                temperature=0,
                verbose=False, # verbose pode ser útil para debug
                openai_api_key=openai_api_key,
                azure_endpoint=azure_endpoint,
                azure_deployment=azure_deployment,
                api_version=api_version,
            )
            print(f"Usando Azure Deployment: {azure_deployment}")

    # Verifica Ollama APENAS se Azure não foi configurado com sucesso
    if llm is None and use_ollama:
        ollama_model = config("OLLAMA_DEFAULT_MODEL", default="llama3.2")
        ollama_api_url = config("OLLAMA_API_URL", default="http://ollama:11434/api/generate") # Lê a URL da config
        print(f"Configurando para Ollama: Modelo={ollama_model}, URL={ollama_api_url}")
        llm = ChatOllama(
            model=ollama_model,
            api_url=ollama_api_url, # Passa a URL para a classe
            temperature=0.0 # Pode configurar outros params aqui se necessário
        )

    # Fallback para OpenAI padrão se nenhuma outra opção foi usada/bem-sucedida
    if llm is None:
        print("Configurando para OpenAI padrão...")
        if not openai_api_key:
             print("AVISO: OPENAI_API_KEY não configurada para fallback OpenAI.")
             # Lançar erro ou retornar None? Lançar erro é mais seguro.
             raise ValueError("Nenhum LLM pôde ser inicializado. Verifique as configurações (Azure, Ollama, OpenAI).")

        default_openai_model = config("OPENAI_DEFAULT_MODEL_NAME", default="gpt-3.5-turbo")
        model_to_use = model_id if model_id and model_id not in bedrock_model_ids else default_openai_model
        print(f"Usando modelo OpenAI: {model_to_use}")
        llm = ChatOpenAI(
            temperature=0,
            verbose=False,
            openai_api_key=openai_api_key,
            model_name=model_to_use,
        )

    print(f"LLM inicializado: {type(llm)}")
    return llm
