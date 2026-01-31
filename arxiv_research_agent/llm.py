"""LLM utilities for the arXiv Research Agent."""

import os
import json
from typing import Dict, List

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI configuration (Responses API)
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2")

# Initialize OpenAI client with Azure endpoint
client = None
if AZURE_OPENAI_ENDPOINT:
    try:
        base_url = f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/v1/"
        
        if AZURE_OPENAI_API_KEY:
            # Use API key authentication
            client = OpenAI(
                base_url=base_url,
                api_key=AZURE_OPENAI_API_KEY,
            )
        else:
            # Use Entra ID (Azure AD) authentication
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default"
            )
            client = OpenAI(
                base_url=base_url,
                api_key=token_provider,
            )
    except Exception as e:
        import warnings
        warnings.warn(f"Failed to initialize OpenAI client: {e}")
else:
    import warnings
    warnings.warn("AZURE_OPENAI_ENDPOINT not set. LLM calls will fail.")

# LLM configuration
DEFAULT_MODEL = AZURE_OPENAI_DEPLOYMENT
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 2000


def call_llm(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    json_output: bool = True,
) -> str:
    """Core LLM API call using Responses API.
    
    Args:
        messages: List of message dictionaries with role and content
        model: The model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        json_output: If True, force structured JSON output
        
    Returns:
        The LLM response content as a string
        
    Raises:
        RuntimeError: If OpenAI client is not initialized
        Exception: If LLM API call fails
    """
    if client is None:
        raise RuntimeError("OpenAI client not initialized. Set AZURE_OPENAI_ENDPOINT (and optionally AZURE_OPENAI_API_KEY or use Entra ID).")
    
    try:
        # Convert messages to input format for Responses API
        # Combine system and user messages into a single input
        input_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in messages
        )
        
        # Build API call parameters
        params = {
            "model": model,
            "input": input_text,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        
        # Add structured JSON output format if requested
        if json_output:
            params["text"] = {"format": {"type": "json_object"}}
        
        response = client.responses.create(**params)
        
        # Extract text from response output
        content = response.output_text
        if content is None:
            raise ValueError("LLM returned empty response")
        return content
    except Exception as e:
        raise Exception(f"LLM API call failed: {str(e)}") from e


def parse_json_response(response: str) -> Dict:
    """Parse JSON from LLM response.
    
    Args:
        response: Raw LLM response string
        
    Returns:
        Parsed JSON as dictionary
    """
    return json.loads(response.strip())
