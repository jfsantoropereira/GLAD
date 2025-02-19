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
        
        # Set up raw token stream logging with colors
        logging.basicConfig(
            level=logging.INFO,
            format='\033[90m%(asctime)s\033[0m - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Log system prompt once at startup
        system = self._generate_system_prompt()
        logger.info("\n\033[94m=== SYSTEM PROMPT ===\033[0m")
        logger.info(system)
        
    def _log_conversation(self, title: str, content: str, color: str = '\033[0m'):
        """Log conversation with colors and formatting."""
        logger.info(f"\n{color}=== {title} ===\033[0m\n{content}\n")
    
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
   <answer>Progress update or results...</answer>
   [More thinking/tools if needed]
   <answer>Final results...</answer>
   </endtask>  # Add ONLY when complete

2. For tasks with multiple steps:
   - Use <thinking> tags to plan each step
   - Execute tools and wait for <result> tags
   - Use <answer> tags to keep user informed
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
<answer>I see the issue. Let me define the function first.</answer>

<thinking>Now I'll implement the improved version...</thinking>
<python>
def better_calculation():
    return 42

result = better_calculation()
print(result)
</python>
<result>42</result>
<answer>Great! I've successfully completed the calculation. The result is 42.</answer>
</endtask>"""

        return prompt
    
    async def get_response(self, message: str, system_prompt: str = None) -> Tuple[str, List[str]]:
        try:
            messages = []
            debug_logs = []
            chronological_context = []  # Track everything in order
            
            # Add context from previous conversation with validation
            for ctx in self.current_context:
                if ctx.get('content') and ctx['content'].strip():
                    messages.append({
                        "role": ctx['role'],
                        "content": ctx['content']
                    })
            
            # Add current user message with timestamp
            timestamp = datetime.now()
            message_with_time = f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
            
            # Print the context window header
            print("\n" + "="*80)
            print(f"CONTEXT WINDOW:")
            print("="*80 + "\n")
            print(f"\033[92mUSER: {message_with_time}\033[0m\n")
            
            # Add user message to chronological context
            chronological_context.append({"role": "user", "content": message_with_time})
            
            # Add the current message to messages for API
            messages.append({"role": "user", "content": message_with_time})
            
            # Let XML processor handle task creation
            task_start = await self.xml_processor.start_task(message_with_time)
            
            # Only append task_start if it's not empty and different from the message
            if task_start and task_start.strip() and task_start != message_with_time:
                messages.append({"role": "user", "content": task_start})

            # Get system prompt
            system = self._generate_system_prompt()
            if system_prompt:
                system = f"{system}\n\n{system_prompt}"

            processed_response = ""
            buffer = ""
            current_thinking = ""
            current_tool_call = ""
            current_tool_result = ""
            
            # Get streaming response from Claude
            try:
                logger.info("Creating message stream...")
                message = self.client.messages.create(
                    model=self.config['model']['name'],
                    max_tokens=self.config['model']['max_tokens'],
                    system=system,
                    messages=messages,
                    stream=True
                )

                logger.info("Starting to process stream...")
                for event in message:
                    if hasattr(event, 'type'):
                        logger.debug(f"Event type: {event.type}")
                    
                    if event.type == "content_block_delta":
                        if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                            content = event.delta.text
                            if content:
                                logger.debug(f"Received content delta: {content}")
                                buffer += content
                                processed_response += content

                                # Process complete tags in buffer
                                result = await self.xml_processor.process_stream_buffer(buffer)
                                
                                # Print any console output from XML processor
                                if result.console_output:
                                    print(result.console_output)
                                    
                                    # Extract and store thinking content
                                    if "<thinking>" in buffer and "</thinking>" in buffer:
                                        thinking_match = re.search(r'<thinking>(.*?)</thinking>', buffer, re.DOTALL)
                                        if thinking_match:
                                            current_thinking = thinking_match.group(1).strip()
                                            chronological_context.append({"role": "assistant", "content": f"<thinking>{current_thinking}</thinking>"})
                                    
                                    # Extract and store tool calls and results
                                    for tool in ['python', 'terminal', 'perplexity']:
                                        if f"<{tool}>" in buffer and f"</{tool}>" in buffer:
                                            tool_match = re.search(f'<{tool}>(.*?)</{tool}>', buffer, re.DOTALL)
                                            if tool_match:
                                                current_tool_call = tool_match.group(1).strip()
                                                chronological_context.append({"role": "assistant", "content": f"<{tool}>{current_tool_call}</{tool}>"})
                                    
                                    # Store tool results
                                    if "<result>" in result.console_output and "</result>" in result.console_output:
                                        result_match = re.search(r'<result>(.*?)</result>', result.console_output, re.DOTALL)
                                        if result_match:
                                            current_tool_result = result_match.group(1).strip()
                                            chronological_context.append({"role": "tool", "content": f"<result>{current_tool_result}</result>"})
                                
                                # Only update buffer and continue if tool execution is complete
                                if buffer == result.remaining_buffer:
                                    continue
                                
                                buffer = result.remaining_buffer
                                
                                # Let the AI determine when to end the task
                                if result.task_complete:
                                    if self.xml_processor.current_task:
                                        task_id = self.xml_processor.current_task
                                        self.xml_processor.active_tasks[task_id].update({
                                            'status': 'completed',
                                            'end_time': datetime.now()
                                        })
                                        self.xml_processor.task_history[task_id] = self.xml_processor.active_tasks[task_id]
                                        self.xml_processor.current_task = None
                                    break

                logger.info(f"Stream processing complete. Response length: {len(processed_response)}")
                if not processed_response:
                    processed_response = "<answer>I apologize, but I wasn't able to generate a proper response. Please try again.</answer>"
                
                # Ensure the response has answer tags
                if "<answer>" not in processed_response:
                    processed_response = f"<answer>{processed_response}</answer>"

                # Add final answer to chronological context
                chronological_context.append({"role": "assistant", "content": processed_response})
                
                # Update the main context with the chronological context
                self.current_context = chronological_context
                
                return processed_response, []
                
            except Exception as e:
                logger.error(f"Streaming error: {str(e)}")
                if "invalid_request_error" in str(e):
                    logger.error(f"Message payload: {messages}")
                raise
            
        except Exception as e:
            print(f"\nERROR: {str(e)}")
            logger.exception("Full exception details:")
            return f"<answer>I encountered an error: {str(e)}</answer>", []
    
    def update_context(self, message: str, response: str):
        """Update conversation context with new message and response."""
        # Add user message to context if not empty
        if message and message.strip():
            self.current_context.append({
                "role": "user",
                "content": message
            })
        
        # Add assistant response to context if not empty
        if response and response.strip():
            self.current_context.append({
                "role": "assistant",
                "content": response
            })
        
        # Trim context if it gets too long while keeping at least the last exchange
        while len(str(self.current_context)) > self.max_context_length and len(self.current_context) > 2:
            self.current_context.pop(0)
            self.current_context.pop(0)  # Remove in pairs to keep context coherent
    
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