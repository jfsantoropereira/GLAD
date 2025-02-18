import asyncio
import sys
import os
import subprocess
import signal
import multiprocessing
from io import StringIO
from typing import Optional, Dict, Any
from .base import BaseToolExecutor

class PythonExecutor(BaseToolExecutor):
    """Executes Python code with safety measures and proper timeout handling."""
    
    def __init__(self):
        super().__init__()
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define restricted builtins for safe execution
        self.safe_builtins = {
            'abs': abs, 'all': all, 'any': any, 'ascii': ascii,
            'bin': bin, 'bool': bool, 'bytearray': bytearray,
            'bytes': bytes, 'chr': chr, 'complex': complex,
            'dict': dict, 'divmod': divmod, 'enumerate': enumerate,
            'filter': filter, 'float': float, 'format': format,
            'frozenset': frozenset, 'hash': hash, 'hex': hex,
            'int': int, 'isinstance': isinstance, 'issubclass': issubclass,
            'iter': iter, 'len': len, 'list': list, 'map': map,
            'max': max, 'min': min, 'next': next, 'oct': oct,
            'ord': ord, 'pow': pow, 'print': print, 'range': range,
            'repr': repr, 'reversed': reversed, 'round': round,
            'set': set, 'slice': slice, 'sorted': sorted, 'str': str,
            'sum': sum, 'tuple': tuple, 'type': type, 'zip': zip
        }
    
    async def _install_package(self, package: str) -> bool:
        """Install a Python package using pip."""
        try:
            # Check if package name is safe (only alphanumeric, -, _, .)
            if not all(c.isalnum() or c in '-_.' for c in package):
                return False
            
            process = await asyncio.create_subprocess_shell(
                f"pip install {package}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            return process.returncode == 0
        except Exception:
            return False
    
    async def _handle_import_error(self, error_msg: str) -> bool:
        """Handle ImportError by attempting to install the missing package."""
        # Extract package name from error message
        if "No module named" in error_msg:
            package = error_msg.split("'")[1]
            return await self._install_package(package)
        return False
    
    def _run_code_in_process(self, code: str, namespace: Dict[str, Any]) -> str:
        """Run code in a separate process with restricted environment."""
        # Capture stdout
        old_stdout = sys.stdout
        redirected_output = StringIO()
        sys.stdout = redirected_output
        
        try:
            # Create restricted globals
            restricted_globals = {
                '__builtins__': self.safe_builtins,
                'os': os,
                'output_dir': self.output_dir
            }
            restricted_globals.update(namespace)
            
            # Execute code
            exec(code, restricted_globals, namespace)
            
            # Get output
            output = redirected_output.getvalue()
            
            # If there's no output but there are new variables, show the last assigned value
            if not output.strip() and namespace:
                # Only show the last value if it's a safe type
                last_var = list(namespace.values())[-1]
                if isinstance(last_var, (int, float, str, list, dict, tuple, set, bool)):
                    output = str(last_var)
            
            return output.strip() if output.strip() else "Code executed successfully (no output)"
            
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            sys.stdout = old_stdout
    
    async def execute(self, content: str) -> str:
        """Execute Python code safely with proper timeout handling."""
        try:
            # Create namespace for execution
            namespace: Dict[str, Any] = {}
            
            # Create a process pool for execution
            with multiprocessing.Pool(1) as pool:
                # Run code in separate process with timeout
                try:
                    async_result = pool.apply_async(self._run_code_in_process, (content, namespace))
                    result = async_result.get(timeout=self.timeout)
                    
                    # Check if result is an error
                    if result.startswith("Error: No module named"):
                        # Try to install missing package and retry
                        if await self._handle_import_error(result[7:]):  # Skip "Error: "
                            async_result = pool.apply_async(self._run_code_in_process, (content, namespace))
                            result = async_result.get(timeout=self.timeout)
                    
                    return self.format_result(result)
                    
                except multiprocessing.TimeoutError:
                    # Terminate the pool forcefully
                    pool.terminate()
                    pool.join()
                    return self.format_error(f"Code execution timed out after {self.timeout} seconds")
                    
                except Exception as e:
                    return self.format_error(str(e))
                
        except Exception as e:
            return self.format_error(f"Failed to execute code: {str(e)}")
        
    def _is_safe_code(self, code: str) -> bool:
        """Check if code appears safe to execute."""
        # List of dangerous patterns
        dangerous_patterns = [
            "import os",
            "import subprocess",
            "import sys",
            "__import__",
            "eval(",
            "exec(",
            "open(",
            "file(",
            ".read(",
            ".write(",
            "os.",
            "sys.",
            "subprocess.",
            "lambda",
            "globals()",
            "locals()"
        ]
        
        # Check for dangerous patterns
        code_lower = code.lower()
        return not any(pattern.lower() in code_lower for pattern in dangerous_patterns) 