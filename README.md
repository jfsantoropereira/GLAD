# GLAD (Global LLM Access Daemon)

GLAD is a system that enables Claude to interact via Telegram and execute various tools, including Python code, terminal commands, and Perplexity searches. It features a simple file-based memory system and XML tag processing for tool instructions.

## Features

- Telegram bot interface for user interaction
- Claude API integration
- Tool execution system (Python, Terminal, Perplexity)
- File-based memory management
- XML tag processing for instructions

## Requirements

- Python 3.11+
- Telegram Bot Token
- Anthropic API Key
- Perplexity API Key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd GLAD
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r txt_files/requirements.txt
```

4. Create a `.env` file in the project root with the following:
```
ANTHROPIC_API_KEY=your_anthropic_api_key
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
PERPLEXITY_API_KEY=your_perplexity_api_key
```

## Usage

Run the main application:
```bash
python main.py
```

## Project Structure

```
GLAD/
├── main.py                      # Main execution loop
├── config/
│   ├── config.json              # Core configuration settings
│   └── platforms.json           # Platform-specific settings
├── system/
│   ├── ai/
│   │   ├── context.py           # LLM context management
│   │   └── memory.py            # Memory management
│   └── platforms/
│       └── telegram.py          # Telegram bot integration
├── tools/
│   ├── executor/
│   │   ├── python_runtime.py    # Python code execution
│   │   └── terminal.py          # Terminal commands
│   └── web/
│       └── perplexity.py        # Perplexity search
└── memory/
    └── long_term.txt            # Persistent storage
```

## Development

To run tests:
```bash
pytest
```

## License

[MIT License](LICENSE)

## Security Note

This MVP version has unrestricted tool access (no sandboxing) and basic security measures. Use in a controlled environment only. 