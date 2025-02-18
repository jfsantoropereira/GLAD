from setuptools import setup, find_packages

setup(
    name="GLAD",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'anthropic',
        'aiogram',
        'openai',
        'python-dotenv',
        'aiohttp'
    ],
) 