import os
import json
import logging
import fcntl
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, memory_file: str = "memory/long_term.txt"):
        self.memory_file = memory_file
        self._ensure_memory_file_exists()
    
    def _ensure_memory_file_exists(self):
        """Ensure the memory file and directory exist."""
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        if not os.path.exists(self.memory_file):
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                f.write('')
    
    async def store_memory(self, content: str, tags: List[str] = None) -> bool:
        """Store a new memory entry with timestamp."""
        try:
            entry = {
                'timestamp': datetime.now().isoformat(),
                'content': content,
                'tags': tags or []
            }
            
            with open(self.memory_file, 'a', encoding='utf-8') as f:
                # Get an exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(entry) + '\n')
                finally:
                    # Release the lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return True
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return False
    
    async def retrieve_memories(self, 
                              tags: List[str] = None, 
                              limit: int = 10, 
                              since: datetime = None) -> List[Dict]:
        """Retrieve memories, optionally filtered by tags and time."""
        memories = []
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                # Get a shared lock for reading
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    for line in f:
                        if line.strip():
                            memory = json.loads(line)
                            
                            # Apply filters
                            if since and datetime.fromisoformat(memory['timestamp']) < since:
                                continue
                                
                            if tags and not all(tag in memory['tags'] for tag in tags):
                                continue
                            
                            memories.append(memory)
                            
                            if len(memories) >= limit:
                                break
                finally:
                    # Release the lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    
            return memories
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []
    
    async def clear_memories(self) -> bool:
        """Clear all stored memories."""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                # Get an exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write('')
                finally:
                    # Release the lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return True
        except Exception as e:
            logger.error(f"Error clearing memories: {e}")
            return False 