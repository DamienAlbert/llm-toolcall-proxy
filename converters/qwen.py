#!/usr/bin/env python3
"""
Qwen model specific tool call converter
"""

import json
import re
from typing import Dict, Any, List
from .base import ToolCallConverter, StreamingToolCallHandler
from config import Config

class QwenToolCallConverter(ToolCallConverter):
    """Converts Qwen tool call format to standard OpenAI format"""
    
    # Qwen model patterns
    QWEN_MODEL_PATTERNS = [
        r'qwen.*'
    ]
    
    def __init__(self):
        """Initialize Qwen converter with config"""
        self.config = Config()
    
    def can_handle_model(self, model_name: str) -> bool:
        """Check if this converter can handle Qwen models"""
        if not model_name:
            return False
            
        model_lower = model_name.lower()
        for pattern in self.QWEN_MODEL_PATTERNS:
            if re.match(pattern, model_lower):
                return True
        return False
    
    def parse_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Parse Qwen format tool calls from content"""
        tool_calls = []
        
        print(f"[DEBUG] Qwen Tool Call Parser - Input content:")
        print(f"[DEBUG] {repr(content)}")
        
        # Regex to capture function blocks directly (works with or without <tool_call> wrapper)
        function_pattern = r'<function=(.*?)>(.*?)</function>'
        function_matches = re.findall(function_pattern, content, re.DOTALL)
        
        print(f"[DEBUG] Found {len(function_matches)} function blocks")
        
        for i, (function_name, params_block) in enumerate(function_matches):
            function_name = function_name.strip()
            
            print(f"[DEBUG] Processing function: {function_name}")
            
            # Parse parameters
            param_pattern = r'<parameter=(.*?)>(.*?)</parameter>'
            param_matches = re.findall(param_pattern, params_block, re.DOTALL)
            
            arguments = {}
            for param_name, param_value in param_matches:
                key = param_name.strip()
                value = param_value.strip()
                print(f"[DEBUG] Param: {key} = {repr(value)}")
            
                # Try to parse JSON values if possible
                try:
                    # Attempt to parse as JSON if it looks like JSON
                    if value.startswith('[') or value.startswith('{'):
                        parsed_value = json.loads(value)
                        arguments[key] = parsed_value
                    else:
                        arguments[key] = value
                except json.JSONDecodeError:
                    # If JSON parsing fails, keep as string
                    arguments[key] = value
                
            # Construct tool call object - note: we want arguments to remain as a dict,
            # which will be serialized properly by the final tool call structure
            tool_call = {
                "id": str(hash(f"{function_name}_{i}_{len(tool_calls)}") % 1000000000),
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False)
                }
            }
            tool_calls.append(tool_call)
            print(f"[DEBUG] Added tool call: {tool_call}")
                
        return tool_calls
    
    def has_partial_tool_call(self, content: str) -> bool:
        """Check if content contains partial Qwen tool call markup"""
        markers = ['<tool_call>', '</tool_call>', '<function=', '</function>', '<parameter=', '</parameter>']
        
        # Check for complete markers
        if any(marker in content for marker in markers):
            return True
            
        # Check for partial markers at the end of content
        for marker in markers:
            for i in range(1, len(marker)):
                if content.endswith(marker[:i]):
                    return True
        return False
    
    def is_complete_tool_call(self, content: str) -> bool:
        """Check if content contains complete Qwen tool call markup"""
        # If we see the start of a wrapper, we MUST wait for the end of the wrapper
        if '<tool_call>' in content:
            return '</tool_call>' in content
            
        # Otherwise, check for complete standalone function
        return bool(re.search(r'<function=.*?</function>', content, re.DOTALL))
    
    def _clean_content(self, content: str) -> str:
        """Remove Qwen tool call markup from content"""
        # Remove wrapped tool calls
        content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL)
        # Remove standalone function calls
        content = re.sub(r'<function=.*?</function>', '', content, flags=re.DOTALL)
        
        # Check for lone tags
        if '<tool_call>' in content and '</tool_call>' not in content:
            content = content.replace('<tool_call>', '')
        if '</tool_call>' in content and '<tool_call>' not in content:
            content = content.replace('</tool_call>', '')
            
        return content.strip()

class QwenStreamingHandler(StreamingToolCallHandler):
    """Qwen-specific streaming tool call handler"""
    
    def __init__(self):
        super().__init__(QwenToolCallConverter())

