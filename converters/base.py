#!/usr/bin/env python3
"""
Base interfaces and abstract classes for tool call converters
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import re


class ToolCallConverter(ABC):
    """Abstract base class for tool call format converters"""
    
    @abstractmethod
    def can_handle_model(self, model_name: str) -> bool:
        """Check if this converter can handle the given model"""
        pass
    
    @abstractmethod
    def parse_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Parse tool calls from content and return in standard OpenAI format"""
        pass
    
    @abstractmethod
    def has_partial_tool_call(self, content: str) -> bool:
        """Check if content contains partial tool call markup"""
        pass
    
    @abstractmethod
    def is_complete_tool_call(self, content: str) -> bool:
        """Check if content contains complete tool call markup"""
        pass
    
    def convert_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert response data if tool calls are detected"""
        if not isinstance(response_data, dict) or 'choices' not in response_data:
            return response_data
        
        modified = False
        for choice in response_data['choices']:
            if 'message' not in choice:
                continue
                
            message = choice['message']
            if 'content' not in message:
                continue
                
            content = message['content']
            if not self.has_partial_tool_call(content):
                # Still apply common cleanup even if no tool calls
                content = self._remove_empty_think_tags(content)
                if content != message['content']:
                    message['content'] = content
                    modified = True
                continue
            
            # Parse tool calls from content
            tool_calls = self.parse_tool_calls(content)
            
            if tool_calls:
                # Remove tool call text from content
                clean_content = self._clean_content(content)
                # Apply common cleanup
                clean_content = self._remove_empty_think_tags(clean_content)
                
                # Update message format
                message['tool_calls'] = tool_calls
                message['content'] = clean_content if clean_content else None
                if (message['content'] is None) or len(message['content'].strip()) == 0:
                    del message['content']
                choice['finish_reason'] = 'tool_calls'
                modified = True
        
        return response_data
    
    def _remove_empty_think_tags(self, content: str) -> str:
        """Remove empty <think></think> tags from content (common cleanup)"""
        if not content:
            return content
            
        # Remove empty think tags with any amount of whitespace inside
        # This handles: <think></think>, <think> </think>, <think>\n</think>, etc.
        content = re.sub(r'<think>\s*</think>', '', content, flags=re.DOTALL)
        
        return content
    
    @abstractmethod
    def _clean_content(self, content: str) -> str:
        """Remove tool call markup from content"""
        pass


class StreamingToolCallHandler(ABC):
    """Abstract base class for streaming tool call handlers"""
    
    def __init__(self, converter: ToolCallConverter):
        self.converter = converter
        self.buffer = ""
        self.tool_call_detected = False
        self.tool_call_complete = False
        self.converted_chunks = []
    
    def process_chunk(self, chunk_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a streaming chunk and handle tool call conversion"""
        if not chunk_data.get('choices') or len(chunk_data['choices']) == 0:
            return chunk_data
        
        choice = chunk_data['choices'][0]
        delta = choice.get('delta', {})
        
        # Accumulate content
        if 'content' in delta and delta['content']:
            self.buffer += delta['content']
            
            # Check for tool call markers
            has_partial = self.converter.has_partial_tool_call(self.buffer)
            
            if has_partial:
                self.tool_call_detected = True
                
                # Check if we have complete tool calls
                if self.converter.is_complete_tool_call(self.buffer):
                    self.tool_call_complete = True
                    return self._convert_to_tool_call_chunk(chunk_data)
                else:
                    # Still accumulating, don't send this chunk
                    return None
            elif self.tool_call_detected and not self.tool_call_complete:
                # Flush the buffer as content.
                self.tool_call_detected = False
                
                # Update the current chunk to include the full accumulated buffer
                if 'content' in delta:
                    delta['content'] = self.buffer
                
                # Fall through to send the chunk
                pass
        
        # pass through if tool calls not detected
        if not self.tool_call_detected:
            self.buffer = ""
            return chunk_data
        
        # If tool call is complete, don't send content chunks
        if self.tool_call_complete:
            return None
            
        return chunk_data
    
    def _convert_to_tool_call_chunk(self, original_chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Convert accumulated content to tool call chunks"""
        tool_calls = self.converter.parse_tool_calls(self.buffer)
        
        if not tool_calls:
            return original_chunk
        
        # Clean content
        clean_content = self.converter._clean_content(self.buffer)
        # Apply common cleanup
        clean_content = self.converter._remove_empty_think_tags(clean_content)
        
        # Create tool call chunk
        chunk = {
            "id": original_chunk.get("id", "chunk"),
            "object": "chat.completion.chunk",
            "created": original_chunk.get("created", int(__import__('time').time())),
            "model": original_chunk.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": clean_content if clean_content else None,
                        "tool_calls": [
                            {
                                "index": i,
                                "id": tool_call["id"],
                                "type": tool_call["type"],
                                "function": tool_call["function"]
                            }
                            for i, tool_call in enumerate(tool_calls)
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ]
        }
        
        return chunk
    
    def finalize(self) -> Optional[Dict[str, Any]]:
        """Finalize streaming and return any remaining chunks"""
        if self.tool_call_detected and not self.tool_call_complete:
            # Handle incomplete tool calls
            return {
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
        return None


class PassThroughConverter(ToolCallConverter):
    """Default converter that doesn't modify anything (pass-through)"""
    
    def can_handle_model(self, model_name: str) -> bool:
        return True  # Can handle any model as fallback
    
    def parse_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        return []  # No conversion
    
    def has_partial_tool_call(self, content: str) -> bool:
        return False  # No tool call detection
    
    def is_complete_tool_call(self, content: str) -> bool:
        return False  # No tool call detection
    
    def _clean_content(self, content: str) -> str:
        # Even pass-through converter should clean empty think tags
        return self._remove_empty_think_tags(content)
