import asyncio
import os
import signal
from typing import List, Optional
from .base import BaseToolExecutor

class TerminalExecutor(BaseToolExecutor):
    """Executes terminal commands with safety measures."""
    
    def __init__(self):
        super().__init__()
        # Define allowed commands and their options
        self.allowed_commands = {
            'ls': ['-l', '-a', '-h', '-t', '-r', '--help'],
            'cat': ['--help'],
            'grep': ['-i', '-n', '-r', '-l', '--help'],
            'pwd': ['--help'],
            'echo': [],  # echo is safe with any args
            'head': ['-n', '--help'],
            'tail': ['-n', '--help'],
            'wc': ['-l', '-w', '-c', '--help'],
            'find': ['-name', '-type', '-size', '--help'],
            'sort': ['-n', '-r', '--help'],
            'uniq': ['-c', '-d', '-u', '--help'],
            'date': ['--help'],
            'df': ['-h', '--help'],
            'du': ['-h', '-s', '--help'],
            'ps': ['aux', '-ef', '--help']
        }
        
        # Define dangerous patterns
        self.dangerous_patterns = [
            ';', '&&', '||', '|',  # Command chaining
            '>', '>>', '<',        # Redirection
            '`', '$(',             # Command substitution
            'rm', 'mv', 'cp',      # File operations
            'chmod', 'chown',      # Permission changes
            'sudo', 'su',          # Privilege escalation
            'wget', 'curl',        # Network access
            'ssh', 'ftp',          # Remote access
            'kill', 'pkill',       # Process management
            'dd', 'mkfs',          # Disk operations
            'mount', 'umount',     # Mount operations
            'apt', 'yum', 'brew'   # Package management
        ]
    
    def _is_safe_command(self, command: str) -> bool:
        """Check if a command is safe to execute."""
        # Split command and arguments
        parts = command.strip().split()
        if not parts:
            return False
        
        # Get base command
        base_cmd = parts[0]
        
        # Check if command is allowed
        if base_cmd not in self.allowed_commands:
            return False
        
        # Check for dangerous patterns
        if any(pattern in command for pattern in self.dangerous_patterns):
            return False
        
        # If command has arguments, check if they're allowed
        if len(parts) > 1:
            allowed_options = self.allowed_commands[base_cmd]
            if allowed_options:  # If empty list, all options are allowed (e.g., echo)
                for arg in parts[1:]:
                    # Skip file/directory arguments
                    if arg.startswith('-'):
                        if arg not in allowed_options:
                            return False
        
        return True
    
    async def execute(self, content: str) -> str:
        """Execute terminal command safely with proper timeout handling."""
        try:
            # Check if command is safe
            if not self._is_safe_command(content):
                return self.format_error(
                    "Command not allowed for security reasons. "
                    "Only basic file and text processing commands are permitted."
                )
            
            # Create subprocess with its own process group
            process = await asyncio.create_subprocess_shell(
                content,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group
            )
            
            try:
                # Wait for process with timeout
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
                
                # Check return code and return appropriate output
                if process.returncode == 0:
                    output = stdout.decode().strip()
                    if not output:
                        return self.format_result("Command executed successfully (no output)")
                    return self.format_result(output)
                else:
                    return self.format_error(stderr.decode().strip())
                    
            except asyncio.TimeoutError:
                # Kill the entire process group
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass  # Process already terminated
                
                return self.format_error(f"Command timed out after {self.timeout} seconds")
                
        except Exception as e:
            return self.format_error(f"Failed to execute command: {str(e)}")
    
    def get_allowed_commands(self) -> List[str]:
        """Return list of allowed commands."""
        return list(self.allowed_commands.keys()) 