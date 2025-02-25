import asyncio 
from .local_models import generate_response_tinyllama   
from .logging_module import setup_logger

logger = setup_logger()

async def generate_scenario():
    INSTRUCTION_PROMPT = """
        🚀 Instruction:  
            Generate a persuasive speaking scenario where the user must convince a skeptical individual, group, or organization about a particular idea, innovation, or approach. The scenario should present a clear challenge, concern, or doubt that the user must address. 🤔  

        🎯 Task:  
            The user must craft a compelling, confident, and persuasive response to change their perspective.  

            Ensure the output follows this exact format:  

            SCENARIO_TEXT = \"\"\"  
        🚀 Scenario:  
            [Describe the skeptical individual, group, or organization and their concerns.]  

        🎯 Your Task:  
            Convince them that [topic/idea] provides a game-changing advantage by offering:  
        ✅ [Advantage 1]  
        ✅ [Advantage 2]  
        ✅ [Advantage 3]  

        Make your argument compelling, confident, and persuasive. 🔥  
        \"\"\"  
        The scenarios should be **diverse, realistic, and engaging**, covering a variety of industries, technologies, social issues, or business challenges. Avoid repetition and ensure each scenario presents a **unique** persuasive challenge.  
    """

    # Generate the model's response when passed this instruction prompt and return that response

    response = await generate_response_tinyllama(INSTRUCTION_PROMPT)
    return response 