import logging
import re
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from tools.executor.python_runtime import PythonExecutor
from tools.executor.terminal import TerminalExecutor
from tools.web.perplexity import PerplexityExecutor
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class StreamResult:
    remaining_buffer: str
    console_output: str
    task_complete: bool

class XMLProcessor:
    """Processes XML tags in Claude's responses and executes corresponding tools."""
    
    def __init__(self):
        # Load config
        with open('config/config.json', 'r') as f:
            self.config = json.load(f)['system']['tools']
        
        # Initialize tool executors
        self.executors = {}
        if self.config['python']['enabled']:
            self.executors['python'] = PythonExecutor()
            self.executors['python'].timeout = self.config['python']['max_execution_time']
            
        if self.config['terminal']['enabled']:
            self.executors['terminal'] = TerminalExecutor()
            self.executors['terminal'].timeout = self.config['terminal']['max_execution_time']
            
        if self.config['perplexity']['enabled']:
            self.executors['perplexity'] = PerplexityExecutor()
            self.executors['perplexity'].timeout = self.config['perplexity']['max_execution_time']
        
        # Define XML tag patterns with named groups for better performance
        self.tag_patterns = {
            'task': re.compile(r'<task(?:\s+id="(?P<id>[^"]*)")?\s*>(?P<content>.*?)</task>', re.DOTALL),
            'thinking': re.compile(r'<thinking>(?P<content>.*?)</thinking>', re.DOTALL),
            'answer': re.compile(r'<answer>(?P<content>.*?)</answer>', re.DOTALL),
            'python': re.compile(r'<python>(?P<content>.*?)</python>', re.DOTALL),
            'terminal': re.compile(r'<terminal>(?P<content>.*?)</terminal>', re.DOTALL),
            'perplexity': re.compile(r'<perplexity>(?P<content>.*?)</perplexity>', re.DOTALL),
            'endtask': re.compile(r'</endtask>', re.DOTALL)
        }
        
        # Track active tasks
        self.active_tasks = {}
        
        # Cache for compiled regex patterns
        self._regex_cache = {}
        
        self.task_history = {}
        self.telegram_handler = None  # Will be set by TelegramBot
        self.current_task = None
    
    def _get_regex(self, pattern: str) -> re.Pattern:
        """Get or compile a regex pattern."""
        if pattern not in self._regex_cache:
            self._regex_cache[pattern] = re.compile(pattern, re.DOTALL)
        return self._regex_cache[pattern]
    
    async def process_response(self, response: str) -> Tuple[str, List[str]]:
        """Process a response from Claude, executing tools and collecting memory entries."""
        processed_response = response
        memory_entries = []
        current_pos = 0
        
        try:
            # First, process task tags to track task context
            for match in self.tag_patterns['task'].finditer(response):
                task_id = match.group('id') or f"task-{len(self.active_tasks)}"
                self.active_tasks[task_id] = {
                    'start_time': datetime.now(),
                    'status': 'running'
                }
            
            # Process tool tags and inject results
            for tool_name in ['python', 'terminal', 'perplexity']:
                if tool_name not in self.executors:
                    continue
                    
                for match in self.tag_patterns[tool_name].finditer(processed_response):
                    content = match.group('content').strip()
                    start, end = match.span()
                    
                    try:
                        result = await self.executors[tool_name].execute(content)
                        # Inject result directly after the tool closing tag
                        processed_response = (
                            processed_response[:end] + 
                            f"\n<result>{result}</result>\n" + 
                            processed_response[end:]
                        )
                    except asyncio.TimeoutError:
                        processed_response = (
                            processed_response[:end] + 
                            f"\n<result>Tool execution timed out after {self.executors[tool_name].timeout} seconds</result>\n" + 
                            processed_response[end:]
                        )
                    except Exception as e:
                        processed_response = (
                            processed_response[:end] + 
                            f"\n<result>Tool execution failed: {str(e)}</result>\n" + 
                            processed_response[end:]
                        )
            
            return processed_response, memory_entries
            
        except Exception as e:
            logger.error(f"Error processing XML tags: {e}")
            return f"<result>Failed to process response: {str(e)}</result>", memory_entries
    
    def extract_final_answer(self, response: str) -> Optional[str]:
        """Extract the final answer from the response."""
        try:
            match = self.tag_patterns['answer'].search(response)
            return match.group('content').strip() if match else response.strip()
        except Exception as e:
            logger.error(f"Error extracting final answer: {e}")
            return None
    
    def get_active_tasks(self) -> Dict[str, Dict]:
        """Return information about currently active tasks."""
        return {
            task_id: task_info 
            for task_id, task_info in self.active_tasks.items() 
            if task_info['status'] == 'running'
        }
    
    def get_task_history(self) -> Dict[str, Dict]:
        """Return information about all tasks, including completed ones."""
        return self.active_tasks.copy()
    
    def set_telegram_handler(self, handler):
        """Set the telegram handler for sending answers."""
        self.telegram_handler = handler
    
    async def process_tool_execution(self, content: str) -> str:
        """Process tool execution tags and return the result."""
        try:
            # Extract tag type and content
            tag_match = re.match(r'<(python|terminal|perplexity|answer|task)>(.*?)</\1>', content, re.DOTALL)
            if not tag_match:
                return "Invalid tag format"
                
            tag_type = tag_match.group(1)
            tag_content = tag_match.group(2).strip()
            
            # Handle task tags
            if tag_type == 'task':
                # Extract task ID if present
                task_id_match = re.search(r'id="([^"]*)"', content)
                task_id = task_id_match.group(1) if task_id_match else f"task-{len(self.task_history) + 1}"
                
                # Only create new task if we're not in one
                if not self.current_task:
                    self.current_task = task_id
                    self.active_tasks[task_id] = {
                        'start_time': datetime.now(),
                        'status': 'running'
                    }
                return f"Started task {task_id}"
            
            # Handle endtask
            if '</endtask>' in content:
                if self.current_task:
                    task_id = self.current_task
                    self.active_tasks[task_id].update({
                        'status': 'completed',
                        'end_time': datetime.now()
                    })
                    self.task_history[task_id] = self.active_tasks[task_id]
                    self.current_task = None
                return "Task completed"
            
            # Handle answer tags
            if tag_type == 'answer':
                if self.telegram_handler:
                    await self.telegram_handler.send_answer(tag_content)
                return tag_content
            
            # Handle tool tags
            if tag_type in self.executors:
                try:
                    return await asyncio.wait_for(
                        self.executors[tag_type].execute(tag_content),
                        timeout=self.config[tag_type]['max_execution_time']
                    )
                except asyncio.TimeoutError:
                    return f"Tool execution timed out after {self.config[tag_type]['max_execution_time']} seconds"
                except Exception as e:
                    return f"Tool execution failed: {str(e)}"
            
            return f"Unhandled tag type: {tag_type}"
            
        except Exception as e:
            logger.error(f"Error processing tag: {e}")
            return f"Error processing tag: {str(e)}"

    async def start_task(self, message: str) -> str:
        """Create a new task with the given message."""
        task_id = f"task-{len(self.task_history) + 1}"
        self.current_task = task_id
        self.active_tasks[task_id] = {
            'start_time': datetime.now(),
            'status': 'running'
        }
        return f'<task id="{task_id}">\n{message}'

    async def process_stream_buffer(self, buffer: str) -> StreamResult:
        """Process the streaming buffer and handle any complete tags."""
        console_output = []
        remaining = buffer
        task_complete = False
        should_pause = False
        tool_execution_complete = True  # Track if tool execution is complete

        # Look for complete tags
        for tag_name, pattern in self.tag_patterns.items():
            match = pattern.search(remaining)
            if match and tag_name != 'endtask':  # Skip processing endtask tags
                # Extract the complete tag
                start, end = match.span()
                tag_content = remaining[start:end]
                
                # Process the tag
                if tag_name == 'thinking':
                    console_output.append(f"\033[94m{tag_content}\033[0m")
                elif tag_name == 'answer':
                    result = await self.process_tool_execution(tag_content)
                    console_output.append(f"\033[92m{tag_content}\033[0m")
                elif tag_name in ['python', 'terminal', 'perplexity']:
                    # Signal that we need to pause token generation
                    should_pause = True
                    tool_execution_complete = False
                    result = await self.process_tool_execution(tag_content)
                    
                    # Start yellow color block
                    console_output.append("\033[93m")
                    # Add the tool call
                    console_output.append(tag_content)
                    
                    # Add the result if any and verify injection
                    if result:
                        result_tag = f"<result>{result}</result>"
                        console_output.append(result_tag)
                        # Verify result is properly injected before continuing
                        if result_tag in console_output[-1]:
                            tool_execution_complete = True
                    
                    # End yellow color block
                    console_output.append("\033[0m")
                
                # Only remove processed tag from buffer if tool execution is complete
                if tool_execution_complete:
                    remaining = remaining[end:]

        # Check if the AI generated an endtask tag (but don't process it)
        if '</endtask>' in buffer:
            task_complete = True
            console_output.append("="*80)

        return StreamResult(
            remaining_buffer=remaining if tool_execution_complete else buffer,  # Return full buffer if tool execution isn't complete
            console_output="\n".join(console_output),
            task_complete=task_complete
        )