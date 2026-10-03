import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from google import genai


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is missing from .env file"
    )


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="EduGenie",
    description="Google Gemini Powered Learning Assistant"
)


# =========================================================
# STATIC FILES
# =========================================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# =========================================================
# TEMPLATES
# =========================================================

templates = Jinja2Templates(
    directory="templates"
)


# =========================================================
# GEMINI CLIENT
# =========================================================

client = genai.Client(
    api_key=api_key
)


# Current Gemini model
MODEL = "gemini-3.8-flash"


# =========================================================
# GEMINI RESPONSE FUNCTION
# =========================================================

def generate_response(mode: str, user_input: str):

    # -----------------------------------------------------
    # PROMPTS
    # -----------------------------------------------------

    prompts = {

        # =================================================
        # ASK AI
        # =================================================

        "ask": f"""
You are EduGenie, an AI learning assistant
for college students.

Answer the student's question clearly and simply.

Instructions:

- Use simple English.
- Explain step by step when necessary.
- Use headings when useful.
- Use bullet points when useful.
- Give examples where appropriate.
- Keep the answer educational.
- Avoid unnecessary complexity.
- Do not provide unrelated information.

Student question:

{user_input}
""",

        # =================================================
        # SUMMARIZE
        # =================================================

        "summary": f"""
You are EduGenie, an educational summarizer.

Summarize the following study material.

Requirements:

- Give a short title.
- Give the main points as bullet points.
- Keep important technical terms.
- Keep important facts.
- Make the summary easy for a college student.
- Do not add unrelated information.

Study material:

{user_input}
""",

        # =================================================
        # QUIZ
        # =================================================

        "quiz": f"""
You are EduGenie, a quiz generator for
college students.

Create 5 multiple-choice questions based on
the following topic.

For every question provide:

1. Question
2. Option A
3. Option B
4. Option C
5. Option D
6. Correct Answer
7. Short Explanation

Make the questions educational and suitable
for college students.

Topic:

{user_input}
""",

        # =================================================
        # SIMPLE EXPLANATION
        # =================================================

        "explain": f"""
You are EduGenie, an educational tutor.

Explain the following topic in very simple
language so that a beginner can understand it.

Use this structure:

### What is it?

### Simple Explanation

### Real-World Example

### Important Points

### Short Summary

Topic:

{user_input}
""",

        # =================================================
        # LEARNING PATH
        # =================================================

        "recommend": f"""
You are EduGenie, a learning path assistant.

The student wants to learn:

{user_input}

Create a simple learning roadmap.

Include:

1. What to learn first
2. What to learn next
3. Important concepts
4. Practice activities
5. Mini project ideas
6. Suggested next topic

Keep the roadmap suitable for a college student
and make it easy to follow.
"""
    }


    # -----------------------------------------------------
    # SELECT PROMPT
    # -----------------------------------------------------

    prompt = prompts.get(
        mode,
        prompts["ask"]
    )


    # -----------------------------------------------------
    # RETRY SETTINGS
    # -----------------------------------------------------

    max_attempts = 3


    # -----------------------------------------------------
    # GEMINI REQUEST
    # -----------------------------------------------------

    for attempt in range(max_attempts):

        try:

            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )


            # ---------------------------------------------
            # CHECK RESPONSE
            # ---------------------------------------------

            if response.text:

                return response.text


            return """
### ⚠️ Empty AI Response

EduGenie received an empty response from Gemini.

Please try again.
"""


        # =================================================
        # ERROR HANDLING
        # =================================================

        except Exception as e:

            error_message = str(e)

            error_upper = error_message.upper()


            # =============================================
            # 429 - QUOTA EXCEEDED
            # =============================================

            if (
                "429" in error_message
                or "RESOURCE_EXHAUSTED" in error_upper
                or "QUOTA" in error_upper
            ):

                return """
### ⚠️ Gemini API Quota Reached

EduGenie has reached the current Gemini API
request limit for this project.

The application itself is working correctly,
but Gemini has temporarily stopped accepting
more requests for the current quota.

Please try again after the quota resets.

---

**EduGenie Status:** ✅ Application working

**Gemini API:** ⚠️ Quota exceeded

**Action:** Try again later
"""


            # =============================================
            # 503 - TEMPORARY SERVER / HIGH DEMAND
            # =============================================

            if (
                "503" in error_message
                or "UNAVAILABLE" in error_upper
                or "HIGH DEMAND" in error_upper
            ):

                if attempt < max_attempts - 1:

                    wait_time = 2 ** attempt

                    time.sleep(wait_time)

                    continue


                return """
### ⚠️ Gemini is Temporarily Busy

EduGenie could not generate the response because
the Gemini AI service is temporarily unavailable.

Please wait a few seconds and try again.
"""


            # =============================================
            # 500 - SERVER ERROR
            # =============================================

            if "500" in error_message:

                if attempt < max_attempts - 1:

                    wait_time = 2 ** attempt

                    time.sleep(wait_time)

                    continue


                return """
### ⚠️ Gemini Server Error

The Gemini service returned a temporary server
error.

Please try again later.
"""


            # =============================================
            # 400 - INVALID REQUEST
            # =============================================

            if (
                "400" in error_message
                or "INVALID_ARGUMENT" in error_upper
            ):

                return """
### ⚠️ Invalid Request

EduGenie could not process this request.

Please try entering a shorter or clearer question
and try again.
"""


            # =============================================
            # 401 / 403 - API KEY PROBLEM
            # =============================================

            if (
                "401" in error_message
                or "403" in error_message
                or "UNAUTHENTICATED" in error_upper
                or "PERMISSION_DENIED" in error_upper
            ):

                return """
### ⚠️ Gemini API Authentication Error

EduGenie could not authenticate with the
Gemini API.

Please check the GEMINI_API_KEY in your
.env file.
"""


            # =============================================
            # OTHER ERRORS
            # =============================================

            return f"""
### ⚠️ EduGenie Error

Something went wrong while generating the answer.

Please try again.

Technical details:

{error_message}
"""


    # -----------------------------------------------------
    # FALLBACK
    # -----------------------------------------------------

    return """
### ⚠️ Please Try Again

EduGenie could not generate a response.

Please try again.
"""


# =========================================================
# HOME PAGE
# =========================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "answer": "",
            "question": "",
            "mode": "ask"
        }
    )


# =========================================================
# ASK / GENERATE RESPONSE
# =========================================================

@app.post(
    "/ask",
    response_class=HTMLResponse
)
async def ask(
    request: Request,
    question: str = Form(...),
    mode: str = Form("ask")
):

    # -----------------------------------------------------
    # CLEAN USER INPUT
    # -----------------------------------------------------

    question = question.strip()


    # -----------------------------------------------------
    # EMPTY INPUT
    # -----------------------------------------------------

    if not question:

        answer = """
### ⚠️ Please Enter Something

Enter a question, topic, or study material
and try again.
"""


    # -----------------------------------------------------
    # GENERATE AI RESPONSE
    # -----------------------------------------------------

    else:

        try:

            answer = generate_response(
                mode,
                question
            )

        except Exception as e:

            answer = f"""
### ⚠️ Unexpected Error

Something went wrong while processing
your request.

Technical details:

{str(e)}
"""


    # -----------------------------------------------------
    # RETURN PAGE
    # -----------------------------------------------------

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "answer": answer,
            "question": question,
            "mode": mode
        }
    )