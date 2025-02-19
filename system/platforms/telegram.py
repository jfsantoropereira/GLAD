import os
import json
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from ..ai.context import LLMContext
from ..ai.memory import MemoryManager
from datetime import datetime
import re
from typing import Tuple
import asyncio

logger = logging.getLogger(__name__)

class TelegramBot:
    def __init__(self, llm_context: LLMContext, memory_manager: MemoryManager):
        self.bot = Bot(token=os.getenv('TELEGRAM_BOT_TOKEN'))
        self.dp = Dispatcher()
        self.llm_context = llm_context
        self.memory_manager = memory_manager
        self.current_message = None
        
        # Load platform config
        with open('config/platforms.json', 'r') as f:
            self.config = json.load(f)['telegram']
        
        # Load system config for tool status
        with open('config/config.json', 'r') as f:
            self.system_config = json.load(f)['system']
        
        # Register with XML processor for answer handling
        self.llm_context.xml_processor.set_telegram_handler(self)
        
        # Register handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register message and command handlers."""
        # Command handlers
        self.dp.message.register(self.cmd_start, Command('start'))
        self.dp.message.register(self.cmd_help, Command('help'))
        self.dp.message.register(self.cmd_status, Command('status'))
        self.dp.message.register(self.cmd_clear, Command('clear'))
        self.dp.message.register(self.cmd_tasks, Command('tasks'))
        
        # General message handler
        self.dp.message.register(self.handle_message)
    
    async def cmd_start(self, message: types.Message):
        """Handle /start command."""
        welcome_text = (
            "👋 Hello! I'm GLAD, your AI assistant powered by Claude.\n\n"
            "I can help you with various tasks including:\n"
            "• Running Python code\n"
            "• Executing terminal commands\n"
            "• Performing web searches\n\n"
            "I maintain context of our conversation and can store important information.\n\n"
            "Type /help to see available commands."
        )
        await message.reply(welcome_text)
    
    async def cmd_help(self, message: types.Message):
        """Handle /help command."""
        commands_text = "Available commands:\n\n"
        for cmd, desc in self.config['commands'].items():
            commands_text += f"/{cmd} - {desc}\n"
        
        commands_text += "\nYou can also:\n"
        
        # Only show enabled tools
        if self.system_config['tools']['python']['enabled']:
            commands_text += "• Run Python code by sending it to me\n"
        if self.system_config['tools']['terminal']['enabled']:
            commands_text += "• Execute terminal commands\n"
        if self.system_config['tools']['perplexity']['enabled']:
            commands_text += "• Ask me to search the web\n"
        
        commands_text += "\nTask Management:\n"
        commands_text += "• Use /tasks to see active and recent tasks\n"
        
        await message.reply(commands_text)
    
    async def cmd_status(self, message: types.Message):
        """Handle /status command."""
        memories = await self.memory_manager.retrieve_memories(limit=1)
        context_size = len(self.llm_context.get_current_context())
        active_tasks = self.llm_context.get_active_tasks()
        
        status_text = (
            "📊 System Status:\n\n"
            f"Context Size: {context_size} messages\n"
            f"Memory Entries: {len(memories)} entries\n"
            f"Active Tasks: {len(active_tasks)}\n"
            "System: Operational\n\n"
            "Available Tools:\n"
        )
        
        # Add tool status with emojis
        tools_config = self.system_config['tools']
        for tool_name, config in tools_config.items():
            status_emoji = "✅" if config['enabled'] else "❌"
            timeout = config['max_execution_time']
            status_text += f"{status_emoji} {tool_name.capitalize()}"
            if config['enabled']:
                status_text += f" (timeout: {timeout}s)"
            status_text += "\n"
        
        await message.reply(status_text)
    
    async def cmd_tasks(self, message: types.Message):
        """Handle /tasks command to show task status."""
        active_tasks = self.llm_context.get_active_tasks()
        task_history = self.llm_context.get_task_history()
        
        if not task_history:
            await message.reply("No tasks have been executed yet.")
            return
        
        task_text = "🔄 Active Tasks:\n\n"
        
        # Show active tasks first
        if active_tasks:
            for task_id, task_info in active_tasks.items():
                start_time = task_info['start_time']
                elapsed = (datetime.now() - datetime.fromisoformat(str(start_time))).total_seconds()
                task_text += f"• {task_id} (running for {int(elapsed)}s)\n"
        else:
            task_text += "No active tasks\n"
        
        # Show recently completed tasks
        task_text += "\n✅ Recently Completed Tasks:\n\n"
        completed_tasks = {
            task_id: info for task_id, info in task_history.items()
            if info['status'] == 'completed'
        }
        
        if completed_tasks:
            for task_id, task_info in list(completed_tasks.items())[-5:]:  # Show last 5
                start_time = task_info['start_time']
                end_time = task_info.get('end_time', datetime.now())
                duration = (datetime.fromisoformat(str(end_time)) - 
                          datetime.fromisoformat(str(start_time))).total_seconds()
                task_text += f"• {task_id} (completed in {int(duration)}s)\n"
        else:
            task_text += "No completed tasks\n"
        
        await message.reply(task_text)
    
    async def cmd_clear(self, message: types.Message):
        """Handle /clear command."""
        self.llm_context.clear_context()
        await message.reply("🧹 Conversation context has been cleared.")
    
    async def send_answer(self, content: str):
        """Send answer content to current telegram chat."""
        if self.current_message and content.strip():
            await self._send_message_chunks(self.current_message, content.strip())
    
    async def _send_message_chunks(self, message: types.Message, text: str) -> None:
        """Send a message in chunks if it's too long."""
        if len(text) > self.config['max_message_length']:
            chunks = [text[i:i + self.config['max_message_length']]
                     for i in range(0, len(text), self.config['max_message_length'])]
            for chunk in chunks:
                await message.reply(chunk)
        else:
            await message.reply(text)
    
    async def handle_message(self, message: types.Message):
        """Handle regular messages."""
        try:
            # Store current message for log handler
            self.current_message = message
            
            # Get response from Claude
            response, memory_entries = await self.llm_context.get_response(message.text)
            
            # Store memory entries if any
            for entry in memory_entries:
                await self.memory_manager.store_memory(
                    content=entry,
                    tags=['conversation_memory']
                )
            
            # Store user message in memory
            await self.memory_manager.store_memory(
                content=message.text,
                tags=['user_message']
            )
            
            # Update context with processed response
            self.llm_context.update_context(message.text, response)
            
            # Clear current message
            self.current_message = None
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            logger.exception("Full exception details:")
            if self.current_message:
                await self.current_message.reply(
                    "😕 Sorry, I encountered an error processing your message. "
                    "Please try again later."
                )
            self.current_message = None
    
    async def start(self):
        """Start the bot."""
        try:
            logger.info("Starting Telegram bot...")
            await self.dp.start_polling(self.bot)
        except Exception as e:
            logger.error(f"Failed to start Telegram bot: {e}")
            raise 