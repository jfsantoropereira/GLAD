import logging
import json
from ..ai.context import LLMContext
from ..ai.memory import MemoryManager
from datetime import datetime

logger = logging.getLogger(__name__)

class ConsoleHandler:
    def __init__(self, llm_context: LLMContext, memory_manager: MemoryManager):
        self.llm_context = llm_context
        self.memory_manager = memory_manager
        
        # Load system config for tool status
        with open('config/config.json', 'r') as f:
            self.system_config = json.load(f)['system']
        
        # Register with XML processor for answer handling
        self.llm_context.xml_processor.set_telegram_handler(self)  # We'll reuse telegram handler for consistency
    
    def show_help(self):
        """Show available commands and features."""
        commands_text = "Available commands:\n\n"
        commands_text += "/help - Show this help message\n"
        commands_text += "/status - Show system status\n"
        commands_text += "/clear - Clear conversation context\n"
        commands_text += "/tasks - Show active and recent tasks\n"
        commands_text += "/exit - Exit the program\n\n"
        
        commands_text += "You can also:\n"
        
        # Only show enabled tools
        if self.system_config['tools']['python']['enabled']:
            commands_text += "• Run Python code\n"
        if self.system_config['tools']['terminal']['enabled']:
            commands_text += "• Execute terminal commands\n"
        if self.system_config['tools']['perplexity']['enabled']:
            commands_text += "• Ask to search the web\n"
        
        print(commands_text)
    
    def show_status(self):
        """Show system status."""
        context_size = len(self.llm_context.get_current_context())
        active_tasks = self.llm_context.get_active_tasks()
        
        status_text = (
            "📊 System Status:\n\n"
            f"Context Size: {context_size} messages\n"
            f"Active Tasks: {len(active_tasks)}\n"
            "System: Operational\n\n"
            "Available Tools:\n"
        )
        
        # Add tool status with indicators
        tools_config = self.system_config['tools']
        for tool_name, config in tools_config.items():
            status_indicator = "[✓]" if config['enabled'] else "[✗]"
            timeout = config['max_execution_time']
            status_text += f"{status_indicator} {tool_name.capitalize()}"
            if config['enabled']:
                status_text += f" (timeout: {timeout}s)"
            status_text += "\n"
        
        print(status_text)
    
    def show_tasks(self):
        """Show task status."""
        active_tasks = self.llm_context.get_active_tasks()
        task_history = self.llm_context.get_task_history()
        
        if not task_history:
            print("No tasks have been executed yet.")
            return
        
        print("🔄 Active Tasks:\n")
        
        # Show active tasks first
        if active_tasks:
            for task_id, task_info in active_tasks.items():
                start_time = task_info['start_time']
                elapsed = (datetime.now() - datetime.fromisoformat(str(start_time))).total_seconds()
                print(f"• {task_id} (running for {int(elapsed)}s)")
        else:
            print("No active tasks")
        
        print("\n✅ Recently Completed Tasks:\n")
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
                print(f"• {task_id} (completed in {int(duration)}s)")
        else:
            print("No completed tasks")
    
    def clear_context(self):
        """Clear conversation context."""
        self.llm_context.clear_context()
        print("🧹 Conversation context has been cleared.")
    
    async def send_answer(self, content: str):
        """Print answer content to console."""
        if content.strip():
            print("\nGLAD:", content.strip(), "\n")
    
    async def handle_message(self, message: str):
        """Handle user input."""
        try:
            if message.startswith('/'):
                command = message[1:].lower()
                if command == 'help':
                    self.show_help()
                elif command == 'status':
                    self.show_status()
                elif command == 'clear':
                    self.clear_context()
                elif command == 'tasks':
                    self.show_tasks()
                elif command == 'exit':
                    return False
                else:
                    print("Unknown command. Type /help to see available commands.")
            else:
                # Get response from Claude
                response, memory_entries = await self.llm_context.get_response(message)
                
                # Store memory entries if any
                for entry in memory_entries:
                    await self.memory_manager.store_memory(
                        content=entry,
                        tags=['conversation_memory']
                    )
                
                # Store user message in memory
                await self.memory_manager.store_memory(
                    content=message,
                    tags=['user_message']
                )
                
                # Update context with processed response
                self.llm_context.update_context(message, response)
            
            return True
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            logger.exception("Full exception details:")
            print("\n😕 Sorry, I encountered an error processing your message. Please try again.")
            return True
    
    async def start(self):
        """Start the console interface."""
        try:
            print("\n👋 Welcome to GLAD Console Interface!")
            print("Type /help to see available commands or just start chatting.\n")
            
            running = True
            while running:
                user_input = input("You: ").strip()
                if user_input:
                    running = await self.handle_message(user_input)
                
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
        except Exception as e:
            logger.error(f"Failed to run console interface: {e}")
            raise 