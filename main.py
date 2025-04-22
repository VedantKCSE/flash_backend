from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, List
from pydantic import RootModel

load_dotenv()

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Define a root model where the input is a dictionary like { "topic": [content lines] }
class AssessmentRequest(RootModel):
    root: Dict[str, List[str]]

class FlashcardRequest(BaseModel):
    topic: str
    num_cards: int = 5

@app.post("/flashcards")
def generate_flashcards(req: FlashcardRequest):
    prompt = f"""
    Create {req.num_cards} flashcards on the topic: "{req.topic}".
    Return the result as a JSON array. Each flashcard should be an object with:
    - "question": the question text
    - "answer": the answer text

    Example:
    [
      {{
        "question": "What is phishing?",
        "answer": "Phishing is a cyberattack where attackers trick users into revealing sensitive information."
      }}
    ]
    Only return valid JSON, nothing else.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content
        flashcards = json.loads(content)
        return {"flashcards": flashcards}
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON response from OpenAI.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ✅ NEW ENDPOINT: /learn
class LearnRequest(BaseModel):
    topic: str
    num_cards: int = 5
    bio: str = ""  # Optional field for user's bio

@app.post("/learn")
def generate_learning_cards(req: LearnRequest):
    # Construct the prompt
    prompt = f"""
You are an expert teacher helping a student with the topic "{req.topic}".
The student has provided their bio: "{req.bio}". Use this information to personalize the learning cards slightly if relevant.

Generate {req.num_cards} concise, clear, and beginner-friendly learning points about this topic.
Each point should focus on a key idea, fact, or insight that helps build a strong foundation.

The last learning card should include a simple example explaining the topic in an easy-to-understand way. Make sure the example is clear, relevant, and simple for a beginner to grasp.

Make the lines simple, engaging, and easy to remember — like flashcard content or bullet points in a quick guide.

Return only a valid JSON array of strings like this:
[
  "Concept 1...",
  "Concept 2...",
  "Example: [Insert simple example here]"
]

Do NOT include any explanations or additional formatting — only the clean JSON array.
"""


    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content
        cards = json.loads(content)

        # Ensure the last card is an example
        if len(cards) > 0 and "Example" not in cards[-1]:
            cards[-1] = f"Example: {req.topic} in practice."

        # Clean up topic for response
        safe_topic = req.topic.strip().lower().replace(" ", "_")
        return {safe_topic: cards}

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON response from OpenAI.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/assessment")
def generate_assessment(req: AssessmentRequest):
    try:
        # Extract topic and content
        topic, content_lines = next(iter(req.root.items()))
        content = "\n".join(content_lines)

        # Filter out any content that contains "Example" in the last card (if it exists)
        content_lines = content.split("\n")
        if content_lines[-1].startswith("Example:"):
            content_lines = content_lines[:-1]  # Remove the last line which is an example

        # Rejoin content after filtering out the example
        filtered_content = "\n".join(content_lines)

        # Create the prompt for generating MCQs
        prompt = f"""
You are a smart assessment generator.

Based on the topic "{topic}" and the following content:
\"\"\"
{filtered_content}
\"\"\"

Create a multiple-choice quiz with 3-5 questions. Each question must include:
- "question": the question text
- "options": an object with 4 labeled options: "A", "B", "C", and "D"
- "answer": the correct option label (just "A", "B", "C", or "D")

Respond only in this valid JSON format:
{{
  "topic": "{topic}",
  "questions": [
    {{
      "question": "What is ...?",
      "options": {{
        "A": "Option A",
        "B": "Option B",
        "C": "Option C",
        "D": "Option D"
      }},
      "answer": "C"
    }}
  ]
}}

Do not include any explanations, context, or markdown. Only return clean JSON.
"""

        # Request the quiz from the OpenAI model
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content
        quiz = json.loads(content)

        return quiz

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON response from OpenAI.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
