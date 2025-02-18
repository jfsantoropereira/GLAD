import os
import time
import json
import asyncio
from typing import Dict, Optional
from openai import OpenAI, AsyncOpenAI
from ..executor.base import BaseToolExecutor
import logging

logger = logging.getLogger(__name__)

class PerplexityExecutor(BaseToolExecutor):
    """Executes Perplexity search queries with rate limiting."""
    
    def __init__(self):
        super().__init__()
        # Load config
        with open('config/config.json', 'r') as f:
            config = json.load(f)['system']['tools']['perplexity']
            
        self.timeout = config['max_execution_time']
        self.api_key = os.getenv('PERPLEXITY_API_KEY')
        self.model = config.get('model', 'sonar-pro')
        self.max_tokens = config.get('max_tokens', 300)
        self.temperature = config.get('temperature', 0.7)
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.perplexity.ai",
            timeout=self.timeout
        )
        
        # Rate limiting settings
        self.requests_per_minute = 10
        self.min_request_interval = 60 / self.requests_per_minute
        self.last_request_time = 0
        self.request_times = []
        
        # Error handling settings
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
        # Cache settings
        self.cache: Dict[str, Dict] = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def _wait_for_rate_limit(self):
        """Wait if necessary to comply with rate limits."""
        current_time = time.time()
        
        # Clean up old request times
        self.request_times = [t for t in self.request_times 
                            if current_time - t < 60]
        
        # Check if we've hit the rate limit
        if len(self.request_times) >= self.requests_per_minute:
            # Wait until the oldest request is more than a minute old
            wait_time = 60 - (current_time - self.request_times[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        # Check minimum interval
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
    
    def _get_cached_result(self, query: str) -> Optional[Dict]:
        """Get cached result if available and not expired."""
        if query in self.cache:
            cache_time, result = self.cache[query]
            if time.time() - cache_time < self.cache_ttl:
                return result
            else:
                del self.cache[query]
        return None
    
    def _cache_result(self, query: str, result: Dict):
        """Cache a search result."""
        self.cache[query] = (time.time(), result)
        
        # Clean up old cache entries
        current_time = time.time()
        expired_keys = [k for k, (t, _) in self.cache.items()
                       if current_time - t > self.cache_ttl]
        for k in expired_keys:
            del self.cache[k]
    
    async def execute(self, content: str) -> str:
        """Execute Perplexity search with rate limiting and caching."""
        logger.info(f"Executing perplexity search with query: {content}")
        
        if not self.api_key:
            return "<result>Perplexity API key not configured. Please set the PERPLEXITY_API_KEY environment variable.</result>"
        
        # Check cache first
        cached_result = self._get_cached_result(content)
        if cached_result:
            formatted = self._format_result(cached_result)
            logger.info(f"Returning cached result: {formatted}")
            return f"<result>{formatted}\n(cached result)</result>"
        
        # Initialize retry counter
        retries = 0
        last_error = None
        
        # Prepare messages for the API call
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant that provides accurate, "
                    "detailed, and well-researched answers to questions."
                )
            },
            {
                "role": "user",
                "content": content
            }
        ]
        
        while retries < self.max_retries:
            try:
                # Wait for rate limit
                await self._wait_for_rate_limit()
                
                # Update request tracking
                current_time = time.time()
                self.last_request_time = current_time
                self.request_times.append(current_time)
                
                # Make request using OpenAI client
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                # Process response
                if response.choices and response.choices[0].message:
                    result = {
                        'answer': response.choices[0].message.content,
                        'sources': []  # Perplexity API might provide sources differently
                    }
                    
                    # Cache successful result
                    self._cache_result(content, result)
                    formatted = self._format_result(result)
                    logger.info(f"Returning result: {formatted}")
                    return f"<result>{formatted}</result>"
                else:
                    last_error = "No response content received"
                    
            except Exception as e:
                last_error = str(e)
                if "rate limit" in last_error.lower():
                    await asyncio.sleep(self.retry_delay * (retries + 1))
                
            retries += 1
            if retries < self.max_retries:
                await asyncio.sleep(self.retry_delay * retries)
        
        return f"<result>Search failed after {retries} attempts. Last error: {last_error}</result>"
    
    def _format_result(self, result: dict) -> str:
        """Format the search results into a readable string."""
        logger.info(f"Formatting result: {result}")
        try:
            answer = result.get('answer', '').strip()
            sources = result.get('sources', [])
            
            if not answer:
                logger.warning("No answer found in result")
                return "No answer found for the query."
            
            formatted = f"{answer}\n\n"
            logger.info(f"Formatted result: {formatted}")
            
            if sources:
                formatted += "Sources:\n"
                for i, source in enumerate(sources[:5], 1):  # Limit to top 5 sources
                    title = source.get('title', 'Untitled').strip()
                    url = source.get('url', 'No URL').strip()
                    if title and url:
                        formatted += f"{i}. {title}\n   {url}\n"
            
            return formatted
            
        except Exception as e:
            return f"Error formatting results: {str(e)}"

# Add test function if running directly
if __name__ == "__main__":
    async def test_perplexity():
        executor = PerplexityExecutor()
        result = await executor.execute("What is the current population of Earth?")
        print(result)

    asyncio.run(test_perplexity()) 