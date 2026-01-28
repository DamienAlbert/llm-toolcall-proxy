#!/usr/bin/env python3
"""
Configuration settings for the proxy server
"""

import os
from typing import Dict, Any, Optional

# Try to import python-dotenv for .env file support
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
except ImportError:
    # If python-dotenv is not installed, continue without it
    pass


class Config:
    """Configuration class for proxy server settings"""
    
    def __init__(self):
        # Backend server settings
        self.BACKEND_HOST = os.getenv('BACKEND_HOST', 'localhost')
        self.BACKEND_PORT = int(os.getenv('BACKEND_PORT', '8888'))
        self.BACKEND_PROTOCOL = os.getenv('BACKEND_PROTOCOL', 'http')
        
        # Proxy server settings
        self.PROXY_HOST = os.getenv('PROXY_HOST', '0.0.0.0')
        self.PROXY_PORT = int(os.getenv('PROXY_PORT', '5000'))
        
        # Request timeout settings
        self.REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '3600'))  # Default 60 minutes
        
        # Streaming timeout settings
        self.STREAMING_TIMEOUT = os.getenv('STREAMING_TIMEOUT', '3600')  # Default 60 minutes
        # Set to 'None' or '0' to disable timeout for streaming
        if self.STREAMING_TIMEOUT.lower() in ['none', 'null', '0', 'false']:
            self.STREAMING_TIMEOUT = None
        else:
            self.STREAMING_TIMEOUT = int(self.STREAMING_TIMEOUT)
        
        # Debug settings
        self.DEBUG = os.getenv('DEBUG', 'true').lower() == 'true'
        
        # Tool call conversion settings
        self.ENABLE_TOOL_CALL_CONVERSION = os.getenv('ENABLE_TOOL_CALL_CONVERSION', 'true').lower() == 'true'
        
        # Content filtering settings
        self.REMOVE_THINK_TAGS = os.getenv('REMOVE_THINK_TAGS', 'true').lower() == 'true'
        
        # Logging settings
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    @property
    def backend_url(self) -> str:
        """Get the complete backend URL"""
        return f"{self.BACKEND_PROTOCOL}://{self.BACKEND_HOST}:{self.BACKEND_PORT}"
    
    def get_backend_config(self) -> Dict[str, Any]:
        """Get backend configuration as dictionary"""
        return {
            'host': self.BACKEND_HOST,
            'port': self.BACKEND_PORT,
            'protocol': self.BACKEND_PROTOCOL,
            'url': self.backend_url,
            'timeout': self.REQUEST_TIMEOUT
        }
    
    def get_proxy_config(self) -> Dict[str, Any]:
        """Get proxy configuration as dictionary"""
        return {
            'host': self.PROXY_HOST,
            'port': self.PROXY_PORT,
            'debug': self.DEBUG
        }
    
    def update_from_env(self):
        """Reload configuration from environment variables"""
        self.__init__()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'backend': self.get_backend_config(),
            'proxy': self.get_proxy_config(),
            'features': {
                'tool_call_conversion': self.ENABLE_TOOL_CALL_CONVERSION
            },
            'logging': {
                'level': self.LOG_LEVEL
            }
        }


class DevelopmentConfig(Config):
    """Development configuration with additional debug features"""
    
    def __init__(self):
        super().__init__()
        self.DEBUG = True
        self.LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """Production configuration with optimized settings"""
    
    def __init__(self):
        super().__init__()
        self.DEBUG = False
        self.LOG_LEVEL = 'WARNING'
        self.REQUEST_TIMEOUT = 60  # Shorter timeout for production


class TestingConfig(Config):
    """Testing configuration"""
    
    def __init__(self):
        super().__init__()
        self.DEBUG = True
        self.LOG_LEVEL = 'DEBUG'
        self.BACKEND_HOST = 'localhost'
        self.BACKEND_PORT = 8889  # Different port for testing
        self.PROXY_PORT = 5001    # Different port for testing


# Configuration factory
def get_config(env: Optional[str] = None) -> Config:
    """Get configuration based on environment"""
    if env is None:
        env = os.getenv('FLASK_ENV', 'development')
    
    config_map = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig
    }
    
    config_class = config_map.get(env.lower(), DevelopmentConfig)
    return config_class()


# Default configuration instance
config = get_config()


# Common backend configurations for different services
BACKEND_CONFIGS = {
    'lmstudio': {
        'host': 'localhost',
        'port': 8888,
        'protocol': 'http',
        'description': 'LM Studio local server'
    },
    'ollama': {
        'host': 'localhost',
        'port': 11434,
        'protocol': 'http',
        'description': 'Ollama local server'
    },
    'openai': {
        'host': 'api.openai.com',
        'port': 443,
        'protocol': 'https',
        'description': 'OpenAI API'
    },
    'anthropic': {
        'host': 'api.anthropic.com',
        'port': 443,
        'protocol': 'https',
        'description': 'Anthropic Claude API'
    }
}

def get_backend_config(service_name: str) -> Dict[str, Any]:
    """Get predefined backend configuration for a service"""
    return BACKEND_CONFIGS.get(service_name.lower(), BACKEND_CONFIGS['lmstudio'])