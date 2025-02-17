import streamlit as st
from langchain_groq import ChatGroq
from educhain import Educhain, LLMConfig
from dotenv import load_dotenv
import os
from groq import Groq  # Importing Groq's client

# Load environment variables
load_dotenv()

# Set up API keys securely
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ API Key is missing. Please set it in the .env file.")
    st.stop()

# Configure Groq Model
groq = ChatGroq(model="llama-3.3-70b-versatile")
groq_config = LLMConfig(custom_model=groq)

# Initialize Educhain client
client = Educhain(groq_config)

# Initialize Groq's API client
groq_client = Groq()

# Streamlit App
st.title("Smartstudybot.ai")

# Sidebar options
st.sidebar.title("Options")
option = st.sidebar.selectbox("Choose an option", ["Generate Plan", "Generate Quiz"])

if option == "Generate Plan":
    st.header("Generate Study Plan")
    
    topic = st.text_input("Enter the topic for the lesson plan:")
    num_days = st.number_input("Enter number of days", min_value=1, max_value=30, value=5)
    difficulty = st.selectbox("Select difficulty level", ["Easy", "Medium", "Hard"])

    if st.button("Generate Plan"):
        if topic:
            # Step 1: Get raw lesson plan from Educhain
            raw_plan = client.content_engine.generate_lesson_plan(topic=topic)

            # Check the actual attributes of raw_plan
            if hasattr(raw_plan, 'dict'):
                raw_plan_content = raw_plan.dict()  # Convert to dictionary if it's a Pydantic model
            elif hasattr(raw_plan, 'content'):
                raw_plan_content = raw_plan.content  # If content is a direct attribute
            elif hasattr(raw_plan, 'plan'):
                raw_plan_content = raw_plan.plan  # If stored as `plan`
            else:
                raw_plan_content = str(raw_plan)  # Fallback: Convert the object to a string
            
            # Step 2: Refine the plan with Groq
            prompt = f"""
            You are a helpful assistant. Adapt the following lesson plan for {num_days} days at a {difficulty} difficulty level.

            Topic: {topic}
            
            Raw Lesson Plan:
            {raw_plan_content}

            Please structure the plan clearly and ensure it is well-distributed over {num_days} days.
            """
            
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful teaching assistant."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.5,
                max_completion_tokens=1024,
                top_p=1,
                stop=None,
                stream=False,
            )

            # Step 3: Display the formatted lesson plan
            refined_plan = chat_completion.choices[0].message.content
            st.write("### Adapted Lesson Plan")
            st.text(refined_plan)

        else:
            st.warning("Please enter a topic.")

elif option == "Generate Quiz":
    st.header("Generate Quiz")
    
    topic = st.text_input("Enter the topic for the quiz:")
    quiz_type = st.selectbox("Choose the type of quiz", ["Multiple Choice", "True/False", "Short Answer"])
    num_questions = st.number_input("Number of questions", min_value=1, max_value=20, value=5)
    
    if st.button("Generate Quiz"):
        if topic:
            # Generate quiz questions
            questions = client.qna_engine.generate_questions(topic=topic, num=num_questions, question_type=quiz_type)

            st.write("### Quiz Questions")
            for i, question in enumerate(questions.questions, start=1):
                st.write(f"**Question {i}:** {question.question}")
                
                if quiz_type == "Multiple Choice":
                    for j, option in enumerate(question.options, start=1):
                        st.write(f"{j}. {option}")

                st.write(f"**Correct Answer:** {question.answer}")
                st.write(f"**Explanation:** {question.explanation}")
                st.write("---")
        else:
            st.warning("Please enter a topic.")
