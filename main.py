#!/usr/bin/env python3

import os
import logging
import asyncio
from dotenv import load_dotenv
from system.platforms.telegram import TelegramBot
from system.ai.context import LLMContext
from system.ai.memory import MemoryManager
import re
from typing import Tuple, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set specific log levels for different modules
logging.getLogger('system.ai.xml_processor').setLevel(logging.WARNING)
logging.getLogger('system.platforms.telegram').setLevel(logging.INFO)
logging.getLogger('aiogram').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

async def main():
    """Main entry point for the GLAD system."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Initialize core components
        llm_context = LLMContext()
        memory_manager = MemoryManager()
        
        # Initialize platform interfaces
        telegram_bot = TelegramBot(llm_context, memory_manager)
        
        # Start platform interfaces
        logger.info("Starting GLAD system...")
        await telegram_bot.start()
        
    except Exception as e:
        logger.error(f"Failed to start GLAD system: {e}")
        logger.exception("Full exception details:")
        raise

async def process_response(self, response: str) -> Tuple[str, List[str]]:
    processed_response = response
    memory_entries = []
    
    # First check if there are any uncompleted tool executions
    tool_matches = re.finditer(r'<(python|terminal|perplexity)>(.*?)</(python|terminal|perplexity)>', processed_response, re.DOTALL)
    for match in tool_matches:
        if not re.search(r'<result>.*?</result>', processed_response[match.end():]):
            # Tool execution result not found, process it
            result = await self.process_tool_execution(match.group(0))
            processed_response = (
                processed_response[:match.end()] + 
                f"\n<result>{result}</result>\n" + 
                processed_response[match.end():]
            )

if __name__ == "__main__":
    asyncio.run(main()) 