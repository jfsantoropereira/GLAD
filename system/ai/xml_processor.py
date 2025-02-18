import logging
import re
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from tools.executor.python_runtime import PythonExecutor
from tools.executor.terminal import TerminalExecutor
from tools.web.perplexity import PerplexityExecutor

logger = logging.getLogger(__name__)

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
            'memory': re.compile(r'<memory>(?P<content>.*?)</memory>', re.DOTALL),
            'error': re.compile(r'<error>(?P<content>.*?)</error>', re.DOTALL),
            'progress': re.compile(r'<progress>(?P<content>.*?)</progress>', re.DOTALL),
            'update': re.compile(r'<update>(?P<content>.*?)</update>', re.DOTALL),
            'python': re.compile(r'<python>(?P<content>.*?)</python>', re.DOTALL),
            'terminal': re.compile(r'<terminal>(?P<content>.*?)</terminal>', re.DOTALL),
            'perplexity': re.compile(r'<perplexity>(?P<content>.*?)</perplexity>', re.DOTALL),
            'endtask': re.compile(r'</endtask>', re.DOTALL)
        }
        
        # Track active tasks
        self.active_tasks = {}
        
        # Cache for compiled regex patterns
        self._regex_cache = {}
    
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
            
            # Process memory tags
            for match in self.tag_patterns['memory'].finditer(response):
                content = match.group('content').strip()
                memory_entries.append(content)
            
            # Process progress and update tags (pass through)
            for tag_type in ['progress', 'update']:
                for match in self.tag_patterns[tag_type].finditer(processed_response):
                    continue  # These tags are handled by the platform interface
            
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
            
            # Process endtask tags
            for match in self.tag_patterns['endtask'].finditer(processed_response):
                # Find the parent task and mark it as complete
                for task_id in reversed(list(self.active_tasks.keys())):
                    if self.active_tasks[task_id]['status'] == 'running':
                        self.active_tasks[task_id].update({
                            'status': 'completed',
                            'end_time': datetime.now()
                        })
                        break
            
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
    
    async def process_tool_execution(self, tool_content: str) -> str:
        """Process tool execution and return raw result."""
        try:
            # Extract tool type and code
            tool_match = re.match(r'<(python|terminal|perplexity)>(.*?)</\1>', tool_content, re.DOTALL)
            if not tool_match:
                return "Invalid tool format"
            
            tool_type = tool_match.group(1)
            code = tool_match.group(2).strip()
            
            # Execute tool and return raw result
            if tool_type == 'python':
                return await self.executors['python'].execute(code)
            elif tool_type == 'terminal':
                return await self.executors['terminal'].execute(code)
            elif tool_type == 'perplexity':
                return await self.executors['perplexity'].execute(code)
                
        except Exception as e:
            return f"Error executing {tool_type}: {str(e)}" 