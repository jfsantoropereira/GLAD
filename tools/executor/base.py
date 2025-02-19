import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class BaseToolExecutor:
    """Base class for all tool executors."""
    
    def __init__(self):
        self.timeout = 30  # Default timeout in seconds
    
    async def execute(self, content: str) -> str:
        """Execute the tool with the given content.
        
        Args:
            content: The content to execute
            
        Returns:
            str: The execution result wrapped in appropriate XML tags
        """
        raise NotImplementedError()
    
    def format_result(self, result: str) -> str:
        """Format the execution result."""
        return result
    
    def format_error(self, error: str) -> str:
        """Format an error message."""
        return f"<error>{error}</error>" 