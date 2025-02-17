import google.generativeai as genai
import os
from logging_module import setup_logger
from dotenv import load_dotenv
import json
import re  # Import regex

logger = setup_logger()

load_dotenv()  
GEMINI_API_KEY  = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

def analyze_speech(text):
    """Analyzes speech for persuasion, confidence, and tone using Gemini Pro with a strict response format."""
    logger.info(f"Analyzing speech: {text[:50]}...")  # Preview first 50 characters

    try:
        prompt = f"""
        Analyze the following speech and provide a structured response in strict JSON format. 
        Ensure the output **ONLY** follows this exact format:

        {{
          "persuasion_score": (integer between 1-100),
          "tone": (comma-separated adjectives, e.g., "Confident, Assertive"),
          "improvement": (concise improvement suggestion, max 15 words),
          "clarity": (integer between 1-100),
          "word_choice": (description of word choice, e.g., "Clear, Concise"),
          "storytelling": (integer between 1-100),
          "engagement": (integer between 1-100),
          "power_phrases": (integer count of powerful phrases),
        }}

        Do **not** include extra text or explanations. Just return the JSON object.

        Speech: "{text}"
        """

        response = genai.GenerativeModel("gemini-pro").generate_content(prompt)

        # Extract JSON using regex to avoid extra text
        json_match = re.search(r"\{.*\}", response.text, re.DOTALL)
        structured_response = json_match.group(0) if json_match else None

        # Ensure the response is valid JSON
        try:
            result = json.loads(structured_response)
            logger.info("Analysis complete.")
            return result
        except (json.JSONDecodeError, TypeError):
            logger.error(f"Invalid JSON format from Gemini: {response.text}")
            return None

    except Exception as e:
        logger.error(f"Error analyzing speech: {e}")
        return None

if __name__ == "__main__":
    test_text = '''Good morning everyone, and thank you for being here today. I truly believe that innovation is the key to solving many of the challenges we face in today's world. 
    We’ve made incredible progress in the last few years, but there's still so much more to be done. As a team, it’s crucial that we remain committed, stay motivated, 
    and never lose sight of our goals. Together, we can push the boundaries and create something that will truly make a difference. So let's focus on our strengths and work 
    together to bring about real change. Thank you'''
    analysis_result = analyze_speech(test_text)
    
    logger.info(test_text)

    if analysis_result:
        print("🔍 Analysis Result:")
        print(analysis_result)
        logger.info(analysis_result)
    else:
        logger.error("Speech analysis failed.")
