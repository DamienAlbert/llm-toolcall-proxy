#!/usr/bin/env python3
"""
Flask-based proxy server for chat completions API
Supports both streaming and non-streaming requests
Handles tool call format conversion from GLM to standard format
"""

import json
import re
import logging
import time
from typing import Dict, Any, Optional, Generator
from flask import Flask, request, Response, jsonify
import requests

# Import configuration and modular converters
from config import config
from converters.factory import converter_factory

app = Flask(__name__)
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Legacy ToolCallConverter for backward compatibility
# Now replaced by modular converter system
class ToolCallConverter:
    """Legacy converter - now delegates to modular system"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name
        self._converter = None
    
    def _get_converter(self):
        if self._converter is None:
            self._converter = converter_factory.get_converter(self.model_name or "")
        return self._converter
    
    def parse_glm_tool_calls(self, content: str) -> list:
        """Legacy method - delegates to modular converter"""
        return self._get_converter().parse_tool_calls(content)
    
    def has_partial_tool_call(self, content: str) -> bool:
        """Legacy method - delegates to modular converter"""
        return self._get_converter().has_partial_tool_call(content)
    
    def is_complete_tool_call(self, content: str) -> bool:
        """Legacy method - delegates to modular converter"""
        return self._get_converter().is_complete_tool_call(content)
    
    def convert_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy method - delegates to modular converter"""
        # Try to detect model from response if not set
        if not self.model_name:
            detected_model = converter_factory.detect_model_from_response(response_data)
            if detected_model:
                self.model_name = detected_model
                self._converter = None  # Reset to get new converter
        
        return self._get_converter().convert_response(response_data)

# Legacy StreamingToolCallHandler for backward compatibility
class StreamingToolCallHandler:
    """Legacy streaming handler - now delegates to modular system"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name
        self._handler = None
    
    def _get_handler(self):
        if self._handler is None:
            self._handler = converter_factory.get_streaming_handler(self.model_name or "")
        return self._handler
    
    def process_chunk(self, chunk_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Legacy method - delegates to modular handler"""
        # Try to detect model from chunk if not set
        if not self.model_name and chunk_data.get('model'):
            self.model_name = chunk_data['model']
            self._handler = None  # Reset to get new handler
        
        return self._get_handler().process_chunk(chunk_data)
    
    def finalize(self) -> Optional[Dict[str, Any]]:
        """Legacy method - delegates to modular handler"""
        return self._get_handler().finalize()

class ProxyHandler:
    """Handles proxy requests to backend API"""
    
    def __init__(self, backend_url: str = None):
        self.backend_url = backend_url or config.backend_url
        # Model-specific converter will be determined per request
    
    def forward_request(self, path: str, stream: bool = False, convert_tool_calls: bool = False) -> Response:
        """Forward request to backend and handle response"""
        try:
            # Prepare request
            url = f"{self.backend_url}{path}"
            headers = dict(request.headers)
            headers.pop('Host', None)
            
            # Forward request with appropriate timeout
            timeout = config.STREAMING_TIMEOUT if stream else config.REQUEST_TIMEOUT
            
            print(f"[DEBUG] Making {'streaming' if stream else 'regular'} request with timeout: {timeout}")
            
            response = requests.request(
                method=request.method,
                url=url,
                headers=headers,
                data=request.get_data(),
                stream=stream,
                timeout=timeout
            )
            
            if stream:
                return self._handle_streaming_response(response, convert_tool_calls)
            else:
                return self._handle_regular_response(response, convert_tool_calls)
                
        except Exception as e:
            logger.error(f"Proxy error: {e}")
            return jsonify({"error": "Proxy request failed"}), 500
    
    def _handle_regular_response(self, response: requests.Response, convert_tool_calls: bool = False) -> Response:
        """Handle non-streaming response"""
        try:
            # Try to parse as JSON and convert tool calls if requested
            response_data = response.json()
            
            if convert_tool_calls:
                # Detect model and get appropriate converter
                model_name = converter_factory.detect_model_from_response(response_data)
                converter = converter_factory.get_converter(model_name or "")
                converted_data = converter.convert_response(response_data)
            else:
                converted_data = response_data
            
            return Response(
                json.dumps(converted_data),
                status=response.status_code,
                headers=dict(response.headers),
                mimetype='application/json'
            )
        except json.JSONDecodeError:
            # Return as-is if not JSON
            return Response(
                response.content,
                status=response.status_code,
                headers=dict(response.headers)
            )
    
    def _handle_streaming_response(self, response: requests.Response, convert_tool_calls: bool = False) -> Response:
        """Handle streaming response with optional tool call conversion"""
        def generate(ignore_carbon=False):
            if convert_tool_calls:
                handler = None
                
                for line in response.iter_lines(decode_unicode=False):
                    if not line:
                        # Keep empty lines for proper SSE format
                        yield '\n'
                        continue
                    
                    # Decode bytes to UTF-8 string properly
                    try:
                        decoded_line = line.decode('utf-8')
                    except UnicodeDecodeError:
                        # Fallback to latin-1 if UTF-8 fails
                        decoded_line = line.decode('latin-1')
                        
                    if decoded_line.startswith('data: '):
                        data = decoded_line[6:]  # Remove 'data: ' prefix
                        
                        if ignore_carbon and '"carbon": {' in data:
                            continue

                        if data.strip() == '[DONE]':
                            # Finalize handler and send any remaining chunks
                            if handler:
                                final_chunk = handler.finalize()
                                if final_chunk:
                                    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                            yield "data: [DONE]\n\n"
                            break
                        
                        try:
                            chunk_data = json.loads(data)
                            
                            # Initialize handler with model from first chunk
                            if handler is None:
                                model_name = chunk_data.get('model', '')
                                handler = converter_factory.get_streaming_handler(model_name)
                            
                            processed_chunk = handler.process_chunk(chunk_data)
                            
                            if processed_chunk:
                                yield f"data: {json.dumps(processed_chunk, ensure_ascii=False)}\n\n"
                                
                        except json.JSONDecodeError:
                            # Pass through non-JSON data with proper SSE format
                            yield f"{decoded_line}\n\n"
                    else:
                        # Pass through non-data lines (comments, etc.) with proper format
                        yield f"{decoded_line}\n\n"
            else:
                # Simple passthrough for non-tool-call streaming
                # Ensure proper SSE format is maintained
                for line in response.iter_lines(decode_unicode=False):
                    if not line:
                        # Keep empty lines for proper SSE format
                        yield '\n'
                    else:
                        # Decode bytes to UTF-8 string properly
                        try:
                            decoded_line = line.decode('utf-8')
                        except UnicodeDecodeError:
                            # Fallback to latin-1 if UTF-8 fails
                            decoded_line = line.decode('latin-1')
                        
                        if ignore_carbon and '"carbon": {' in decoded_line:
                            continue
                        yield f"{decoded_line}\n\n"
        
        # Set proper headers for SSE streaming
        headers = {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
        }

        return Response(
            generate(ignore_carbon=True),
            status=response.status_code,
            headers=headers,
            mimetype='text/event-stream'
        )

# Initialize proxy handler
# Force reload check
proxy = ProxyHandler()

@app.route('/chat/completions', methods=['POST'])
@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """Handle chat completions requests"""
    try:
        # Check if streaming is requested
        request_data = request.get_json()
        stream = request_data.get('stream', False) if request_data else False
        
        # Check if tool call conversion is enabled
        convert_tools = config.ENABLE_TOOL_CALL_CONVERSION
        
        return proxy.forward_request('/v1/chat/completions', stream=stream, convert_tool_calls=convert_tools)
        
    except Exception as e:
        logger.error(f"Chat completions error: {e}")
        return jsonify({"error": "Request processing failed"}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "chat-proxy"})

@app.route('/v1/models', methods=['GET'])
def models():
    """List available models"""
    return proxy.forward_request('/v1/models')

@app.route('/v1/completions', methods=['POST'])
def completions():
    """Handle text completions requests (non-chat)"""
    try:
        # Check if streaming is requested
        request_data = request.get_json()
        stream = request_data.get('stream', False) if request_data else False
        
        return proxy.forward_request('/v1/completions', stream=stream)
        
    except Exception as e:
        logger.error(f"Completions error: {e}")
        return jsonify({"error": "Request processing failed"}), 500

@app.route('/v1/embeddings', methods=['POST'])
def embeddings():
    """Handle embeddings requests"""
    try:
        return proxy.forward_request('/v1/embeddings')
        
    except Exception as e:
        logger.error(f"Embeddings error: {e}")
        return jsonify({"error": "Request processing failed"}), 500

@app.route('/v1/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
def proxy_v1(path):
    """Proxy all /v1/* requests transparently"""
    return proxy.forward_request(f'/v1/{path}')

@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
def proxy_all(path):
    """Proxy all other requests transparently"""
    return proxy.forward_request(f'/{path}')

if __name__ == '__main__':
    logger.info(f"Starting proxy server on {config.PROXY_HOST}:{config.PROXY_PORT}")
    logger.info(f"Proxying to backend: {config.backend_url}")
    logger.info(f"Tool call conversion: {'Enabled' if config.ENABLE_TOOL_CALL_CONVERSION else 'Disabled'}")
    
    app.run(
        host=config.PROXY_HOST,
        port=config.PROXY_PORT,
        debug=config.DEBUG
    )