import asyncio
import time
import ollama
from .logging_module import setup_logger

logger = setup_logger()

async def generate_response_tinyllama(user_input: str) -> str:
    """
    Generates a response using the locally running TinyLlama model on Ollama.
    Logs the response and the time taken to generate it.
    """

    prompt = f"""
    User: {user_input}  
    Respond
    """

    start_time = time.time()  # Start timing

    response = await asyncio.to_thread(ollama.chat, model="tinyllama", messages=[{"role": "user", "content": prompt}])

    end_time = time.time()  # End timing
    time_taken = end_time - start_time

    logger.info(f"Response Time: {time_taken:.2f} seconds")
    logger.info(f"Response:\n{response['message']['content']}")

    return response["message"]["content"]

async def generate_response_smallLM(user_input: str) -> str:
    """
    Generates a response using the locally running TinyLlama model on Ollama.
    Logs the response and the time taken to generate it.
    """

    prompt = f"""
    User: {user_input}  
    Respond
    """

    start_time = time.time()  # Start timing

    response = await asyncio.to_thread(ollama.chat, model="smalLM", messages=[{"role": "user", "content": prompt}])

    end_time = time.time()  # End timing
    time_taken = end_time - start_time

    logger.info(f"Response Time: {time_taken:.2f} seconds")
    logger.info(f"Response:\n{response['message']['content']}")

    return response["message"]["content"]