import os
import json
import logging
from anthropic import Anthropic
from typing import Tuple, List, Optional, Dict
from .xml_processor import XMLProcessor
from datetime import datetime
import re
import asyncio

logger = logging.getLogger(__name__)

class LLMContext:
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.current_context = []
        self.max_context_length = 4000
        
        # Load config
        with open('config/config.json', 'r') as f:
            self.config = json.load(f)['system']
        
        self.xml_processor = XMLProcessor()
        
        # Set up raw token stream logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s'
        )
        
        # Log system prompt once at startup
        system = self._generate_system_prompt()
        logger.info("\n=== SYSTEM PROMPT ===")
        logger.info(system)
        
    def _log_conversation(self, title: str, content: str):
        """Log conversation with minimal formatting."""
        logger.info(f"\n=== {title} ===\n{content}\n")
    
    def _generate_system_prompt(self) -> str:
        """Generate system prompt based on enabled tools."""
        base_prompt = """You are GLAD, an AI virtual coworker powered by Claude. 

CRITICAL TIMESTAMP INSTRUCTION:
The timestamp next to each user message [YYYY-MM-DD HH:MM:SS] represents the ACTUAL CURRENT TIME when that message was sent.
You MUST use this timestamp for any time-related calculations or queries, NOT any dates from your training data.
Each message's timestamp is the source of truth for the current date and time.

IMPORTANT RULES:
1. Users can ONLY see content within <answer> </answer> tags
2. Each task starts with <task id="task-id"> and ends with </endtask>
3. You can use multiple <answer> tags within a task to:
   - Provide progress updates
   - Explain errors or issues
   - Show intermediate results
4. ONLY use </endtask> when you are completely finished with the task, including:
   - All tool executions are complete
   - All errors are resolved
   - Final results are explained to the user
5. Never assume users can see your thinking process or tool outputs
6. When using tools:
   - Tool execution stops at the closing tag (</python>, </terminal>, </perplexity>)
   - Results are automatically injected after the closing tag as <result>output</result>
   - Wait for tool results before continuing your response
   - Tool results include both successful outputs and error messages
7. The timestamp next to each user message [YYYY-MM-DD HH:MM:SS] is always the correct and current date/time when that message was sent. Use this timestamp for any time-related queries or calculations."""
        
        # Check which tools are enabled
        enabled_tools = []
        if self.config['tools']['python']['enabled']:
            enabled_tools.append(("Python Code Execution", "<python>\n# Your Python code here\n</python>\n<result>Tool output appears here</result>"))
        if self.config['tools']['terminal']['enabled']:
            enabled_tools.append(("Terminal Commands", "<terminal>\n# Your command here\n</terminal>\n<result>Tool output appears here</result>"))
        if self.config['tools']['perplexity']['enabled']:
            enabled_tools.append(("Perplexity Search", "<perplexity>\n# Your search query here\n</perplexity>\n<result>Tool output appears here</result>"))
        
        # If no tools are enabled, return basic prompt
        if not enabled_tools:
            return base_prompt + "\n\nYou are currently operating in chat-only mode without any external tools enabled."
        
        # Add tool documentation
        prompt = base_prompt + "\n\nYou can execute various tools using XML tags. Available tools and their usage:\n\n"
        
        for tool_name, tool_syntax in enabled_tools:
            prompt += f"{tool_name}:\n{tool_syntax}\n\n"
        
        # Add common tags and examples
        prompt += """Available XML tags:
- <thinking>: Express your reasoning process (not visible to user)
- <answer>: Communicate with the user (ONLY content they see)
- <memory>: Store important information for later (not visible to user)
- <progress>: Report task progress (visible to user)
- <update>: Send autonomous updates (visible to user)
- <task id="task-id">: Marks task start with optional ID
- </endtask>: Marks task completion (use ONLY when fully complete)
- <result>: Automatically added after tool execution (contains tool output or errors)

Task Flow Guidelines:
1. Each task follows this pattern:
   <task id="unique-id">  # Start with task tag
   <thinking>Initial approach...</thinking>
   <python>
   # Your code here
   </python>
   <result>Output from the code execution</result>
   <progress>Working on step 1...</progress>
   <answer>Progress update or results...</answer>
   [More thinking/tools if needed]
   <update>Background operation completed...</update>
   <answer>Final results...</answer>
   </endtask>  # Add ONLY when complete

2. For tasks with multiple steps:
   - Use <thinking> tags to plan each step
   - Use <progress> tags to show step transitions
   - Execute tools and wait for <result> tags
   - Use <answer> tags to keep user informed
   - Use <update> tags for background operations
   - Continue until task is fully complete
   - Only then use </endtask>

3. For error handling:
   - Tool errors appear in <result> tags
   - Process errors and explain in <answer> tags
   - Continue working until resolved
   - Only use </endtask> after success

Example of proper multi-step task:
<task id="calculation-001">
<thinking>I'll first try a basic approach...</thinking>
<python>
# First attempt
result = initial_calculation()
print(result)
</python>
<result>Error: NameError: name 'initial_calculation' is not defined</result>
<progress>Initial attempt failed. Analyzing error...</progress>
<answer>I see the issue. Let me define the function first.</answer>

<thinking>Now I'll implement the improved version...</thinking>
<python>
def better_calculation():
    return 42

result = better_calculation()
print(result)
</python>
<result>42</result>
<update>Calculation completed successfully!</update>
<answer>Great! I've successfully completed the calculation. The result is 42.</answer>
</endtask>"""

        return prompt
    
    async def get_response(self, message: str, system_prompt: str = None) -> Tuple[str, List[str]]:
        """Get a response from Claude and process any tool executions."""
        try:
            # Prepare messages for the conversation history
            messages = []
            
            # Add context from previous conversation
            for ctx in self.current_context:
                messages.append({"role": ctx['role'], "content": ctx['content']})
            
            # Add current user message with timestamp
            timestamp = datetime.now()
            message_with_time = f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
            logger.info(f"Processing message with timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            task_id = f"task-{len(self.xml_processor.get_task_history()) + 1}"
            message_with_task = f'<task id="{task_id}">\n{message_with_time}'
            messages.append({"role": "user", "content": message_with_task})
            logger.info(f"\nUSER: {message_with_time}")
            
            # Get system prompt
            system = self._generate_system_prompt()
            if system_prompt:
                system = f"{system}\n\n{system_prompt}"

            processed_response = ""
            memory_entries = []
            task_complete = False
            max_retries = 5
            retry_count = 0
            
            while not task_complete and retry_count < max_retries:
                # Get initial response from Claude
                response = self.client.messages.create(
                    model=self.config['model']['name'],
                    max_tokens=self.config['model']['max_tokens'],
                    system=system,
                    messages=messages
                )
                
                raw_response = response.content[0].text
                current_pos = 0
                
                # Process the response looking for tags
                while current_pos < len(raw_response):
                    # Look for next tag
                    next_tag_match = re.search(r'<(answer|python|terminal|perplexity)>', raw_response[current_pos:])
                    
                    if next_tag_match:
                        tag_type = next_tag_match.group(1)
                        tag_start = current_pos + next_tag_match.start()
                        
                        # Process everything before the tag
                        pre_tag = raw_response[current_pos:tag_start]
                        if pre_tag:
                            processed_response += pre_tag
                            logger.info(pre_tag)
                        
                        if tag_type == 'answer':
                            # Find the end of the answer tag
                            answer_end = raw_response.find('</answer>', tag_start)
                            if answer_end != -1:
                                answer_content = raw_response[tag_start:answer_end + len('</answer>')]
                                processed_response += answer_content
                                # Log answer immediately
                                logger.info(answer_content)
                                current_pos = answer_end + len('</answer>')
                            else:
                                # Incomplete answer tag, wait for more
                                current_pos = tag_start
                                break
                        else:
                            # Handle tool tags
                            tool_pattern = f'<{tag_type}>(.*?)</{tag_type}>'
                            tool_match = re.search(tool_pattern, raw_response[tag_start:], re.DOTALL)
                            if tool_match:
                                tool_content = raw_response[tag_start:tag_start + tool_match.end()]
                                
                                # Execute tool
                                try:
                                    result = await asyncio.wait_for(
                                        self.xml_processor.process_tool_execution(tool_content),
                                        timeout=self.config['timeout'].get(tag_type, 30)
                                    )
                                except asyncio.TimeoutError:
                                    result = "Tool execution timed out"
                                
                                # Add tool content and wrap raw result in result tags
                                processed_response += tool_content + f"\n<result>{result}</result>"
                                logger.info(tool_content)
                                logger.info(f"\n<result>{result}</result>")
                                
                                current_pos = tag_start + tool_match.end()
                                
                                # Update context and get continuation
                                messages.append({
                                    "role": "assistant",
                                    "content": processed_response.rstrip()
                                })
                                response = self.client.messages.create(
                                    model=self.config['model']['name'],
                                    max_tokens=self.config['model']['max_tokens'],
                                    system=system,
                                    messages=messages
                                )
                                raw_response = response.content[0].text.rstrip()
                                current_pos = 0
                                break
                            else:
                                # Incomplete tool tag
                                current_pos = tag_start
                                break
                    else:
                        # No more tags, add remaining content
                        remaining = raw_response[current_pos:]
                        if remaining:
                            processed_response += remaining
                            logger.info(remaining)
                        break
                
                # Process memory entries
                memory_matches = re.finditer(r'<memory>(.*?)</memory>', processed_response, re.DOTALL)
                for match in memory_matches:
                    memory_entries.append(match.group(1).strip())
                
                # Check if task is complete
                if "</endtask>" in processed_response:
                    task_complete = True
                else:
                    messages.append({"role": "assistant", "content": processed_response.strip()})
                    retry_count += 1

            return processed_response.strip(), memory_entries
            
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            error_response = f"<error>Failed to get response: {str(e)}</error>"
            return error_response.strip(), []
    
    def update_context(self, message: str, response: str):
        """Update conversation context with new message and response."""
        self.current_context.append({"role": "user", "content": message})
        self.current_context.append({"role": "assistant", "content": response})
        
        # Trim context if it gets too long
        while len(str(self.current_context)) > self.max_context_length:
            self.current_context.pop(0)
    
    def clear_context(self):
        """Clear the conversation context."""
        self.current_context = []
    
    def get_current_context(self) -> List[Dict[str, str]]:
        """Get the current conversation context."""
        return self.current_context.copy()
    
    def get_active_tasks(self) -> Dict[str, Dict]:
        """Get information about currently active tasks."""
        return self.xml_processor.get_active_tasks()
    
    def get_task_history(self) -> Dict[str, Dict]:
        """Get information about all tasks, including completed ones."""
        return self.xml_processor.get_task_history() 