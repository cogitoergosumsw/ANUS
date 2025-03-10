"""
Google Gemini Model implementation for the ANUS framework.
"""

from typing import Dict, List, Optional, Union, Any
import json
import os
import logging
from pydantic import Field

# Import from the new SDK
from google import genai 
from google.genai import types as genai_types

from anus.models.base.base_model import BaseModel, ToolCall

class GeminiModel(BaseModel):
    """
    Google Gemini API integration for language models.
    Supports Gemini 2.0 models.
    """
    
    provider: str = "gemini"
    model_name: str = "gemini-2.0-flash"  # Default model
    api_key: Optional[str] = Field(default_factory=lambda: os.environ.get("GOOGLE_API_KEY"))
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize the client with our API key
        self.client = genai.Client(api_key=self.api_key)
    
    async def generate(
        self, 
        prompt: str, 
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text based on a prompt using Google Gemini.
        
        Args:
            prompt: The text prompt for generation.
            system_message: Optional system message for the model.
            temperature: Controls randomness in outputs. Overrides instance value if provided.
            max_tokens: Maximum number of tokens to generate. Overrides instance value if provided.
            **kwargs: Additional Gemini-specific parameters.
            
        Returns:
            The generated text response.
        """
        # Create configuration
        config = {}
        
        # Add system instruction if provided
        if system_message:
            config["system_instruction"] = system_message
            
        # Add other parameters
        if temperature is not None:
            config["temperature"] = temperature
        elif self.temperature is not None:
            config["temperature"] = self.temperature
            
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens
        elif self.max_tokens is not None:
            config["max_output_tokens"] = self.max_tokens
        
        try:
            # Generate response
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            
            return response.text
        
        except Exception as e:
            logging.error(f"Error generating with Gemini: {e}")
            return f"Error: {str(e)}"
    
    async def generate_with_tools(
        self, 
        prompt: str, 
        tools: List[Dict[str, Any]],
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text with tool calling capabilities.
        
        Args:
            prompt: The text prompt for generation.
            tools: List of tool schemas available for use.
            system_message: Optional system message for the model.
            temperature: Controls randomness in outputs. Overrides instance value if provided.
            max_tokens: Maximum number of tokens to generate. Overrides instance value if provided.
            **kwargs: Additional Gemini-specific parameters.
            
        Returns:
            A dictionary with the response and any tool calls.
        """
        # Convert our tools to Gemini format
        gemini_tools = []
        for tool in tools:
            function = tool.get("function", {})
            
            # Create function declaration
            function_declaration = genai_types.FunctionDeclaration(
                name=function.get("name", ""),
                description=function.get("description", ""),
                parameters=function.get("parameters", {})
            )
            
            # Add to tools list
            gemini_tools.append(genai_types.Tool(
                function_declarations=[function_declaration]
            ))
        
        # Create configuration
        config = {}
        
        # Add system instruction if provided
        if system_message:
            config["system_instruction"] = system_message
            
        # Add parameters
        if temperature is not None:
            config["temperature"] = temperature
        elif self.temperature is not None:
            config["temperature"] = self.temperature
            
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens
        elif self.max_tokens is not None:
            config["max_output_tokens"] = self.max_tokens
            
        # Add tools and tool choice settings
        if gemini_tools:
            config["tools"] = gemini_tools
            
            # Handle tool choice
            tool_choice = kwargs.get("tool_choice", "auto")
            if tool_choice == "required":
                config["automatic_function_calling"] = {"disable": False}
            elif tool_choice == "none":
                config["automatic_function_calling"] = {"disable": True}
            # "auto" is the default in Gemini
        
        try:
            # Generate response with tools
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            
            # Extract content and tool calls
            content = response.text
            tool_calls = []
            
            # Process tool calls if present
            if hasattr(response, "candidates") and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, "content") and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, "function_call"):
                                # Convert to our ToolCall format
                                tool_calls.append(
                                    ToolCall(
                                        id=str(len(tool_calls)),
                                        type="function",
                                        function={
                                            "name": part.function_call.name,
                                            "arguments": part.function_call.args
                                        }
                                    )
                                )
            
            return {
                "content": content,
                "tool_calls": tool_calls
            }
            
        except Exception as e:
            logging.error(f"Error generating with tools using Gemini: {e}")
            return {
                "content": f"Error: {str(e)}",
                "tool_calls": []
            }
    
    async def extract_json(
        self, 
        prompt: str, 
        schema: Dict[str, Any],
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract structured JSON data based on a prompt.
        
        Args:
            prompt: The text prompt for extraction.
            schema: JSON schema describing the expected structure.
            system_message: Optional system message for the model.
            temperature: Controls randomness in outputs. Overrides instance value if provided.
            max_tokens: Maximum number of tokens to generate. Overrides instance value if provided.
            **kwargs: Additional Gemini-specific parameters.
            
        Returns:
            The extracted JSON data.
        """
        # Create configuration
        config = {
            "response_mime_type": "application/json",
            "response_schema": schema
        }
        
        # Add system instruction if provided
        if system_message:
            config["system_instruction"] = system_message
            
        # Add parameters
        if temperature is not None:
            config["temperature"] = temperature
        elif self.temperature is not None:
            config["temperature"] = self.temperature
            
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens
        elif self.max_tokens is not None:
            config["max_output_tokens"] = self.max_tokens
        
        try:
            # Generate response
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            
            # The SDK may have parsed JSON automatically
            if hasattr(response, "parsed"):
                return response.parsed
            
            # Otherwise try to parse the JSON
            try:
                return json.loads(response.text)
            except json.JSONDecodeError:
                logging.error(f"Failed to parse JSON from response: {response.text}")
                return {"error": "Failed to parse JSON response"}
                
        except Exception as e:
            logging.error(f"Error extracting JSON with Gemini: {e}")
            return {"error": str(e)}
    
    async def get_embedding(self, text: str, **kwargs) -> List[float]:
        """
        Generate an embedding vector for the given text.
        
        Args:
            text: The text to embed.
            **kwargs: Additional Gemini-specific parameters.
            
        Returns:
            The embedding vector as a list of floats.
        """
        try:
            # Generate embeddings using the embedding model
            response = self.client.models.embed_content(
                model="text-embedding-004",  # Default embedding model
                contents=text
            )
            
            # Extract the embedding values
            if hasattr(response, "embedding"):
                return response.embedding.values
            
            return []
            
        except Exception as e:
            logging.error(f"Error generating embedding with Gemini: {e}")
            return []