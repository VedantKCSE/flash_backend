from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel, RootModel
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, List
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

origins = [
    "http://localhost:19006",  # Expo Web
    "http://127.0.0.1:8000",   # FastAPI local
    "http://192.168.x.x:8000", # Your LAN IP, more on this below
    "*"  # for testing only
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Root model: dictionary like { "topic": [content lines] }
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


# ✅ /learn endpoint
class LearnRequest(BaseModel):
    topic: str
    num_cards: int = 5
    bio: str = ""  # Optional

@app.post("/learn")
def generate_learning_cards(req: LearnRequest):
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

        safe_topic = req.topic.strip().lower().replace(" ", "_")
        return {safe_topic: cards}

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON response from OpenAI.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ✅ /assessment endpoint
@app.post("/assessment")
def generate_assessment(req: AssessmentRequest):
    try:
        # Extract topic and content
        topic, content_lines = next(iter(req.root.items()))
        content = "\n".join(content_lines)

        # Convert back to list in case of previous join
        content_lines = content.split("\n")

        # Remove example if present
        if content_lines[-1].startswith("Example:"):
            content_lines = content_lines[:-1]

        filtered_content = "\n".join(content_lines)

        # Determine number of questions based on number of statements
        num_statements = len(content_lines)
        approx_questions = min(num_statements, 15)

        prompt = f"""
You are a smart assessment generator.

Based on the topic "{topic}" and the following content:
\"\"\"
{filtered_content}
\"\"\"

Create a multiple-choice quiz with approximately {approx_questions} questions. Each question must include:
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
