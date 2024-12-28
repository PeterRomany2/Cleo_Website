import ollama
import re
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(layout="wide")

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("You need to log in to access this page.")
    st.stop()  # Stop further execution of the page









model = 'llama3.1'

# Embed Lottie animation in Streamlit with top-left positioning using inline CSS
lottie_html = """

<div style="text-align: center;">
<h1 style="color:#0FFF50;">AI Cleo Assistant<br>Your Way to Better Business Decisions</h1>
</div>
<div style="position: fixed; top: 0; right: 0;">
<script src="https://unpkg.com/@lottiefiles/lottie-player@2.0.8/dist/lottie-player.js"></script><lottie-player src="https://lottie.host/3e8ec22e-eb1b-4a61-8b78-5550fe0b94b8/AK7Vjkbjxk.json" background="##FFFFFF" speed="1" style="width: 300px; height: 300px" loop  autoplay direction="1" mode="normal"></lottie-player>
</div>
"""
# <div style="position: fixed; bottom: 0; left: 0;">
# <script src="https://unpkg.com/@lottiefiles/lottie-player@2.0.8/dist/lottie-player.js"></script><lottie-player src="https://lottie.host/a0ebc27b-5ef0-4da2-8ff3-d04dbc033263/W7H2wmEBC3.json" background="##FFFFFF" speed="1" style="width: 300px; height: 300px" loop  autoplay direction="1" mode="normal"></lottie-player>
# </div>
components.html(lottie_html, height=250)


def chat(message):

    user_message = [{
        'role': 'user',
        'content': fr"""
        use: df = pd.read_csv(r"C:\Users\beter\PycharmProjects\QA-Engine\Cleo_Data.csv")
and use: column names of this df:['DATE', 'SALES', 'ORDERS', 'CUSTOMER_REVIEWS']
your task is: {message} (perform this task in python and show results in streamlit code with plotly)(write only the required code)
"""
    }]

    response = ollama.chat(model=model, messages=[user_message[0]])
    answer = response['message']['content']
    return answer


def execute_code(code):
    try:
        exec(code)

    except SyntaxError as e:
        st.write(f"Syntax error in the generated code: {str(e)}")
    except Exception as e:
        st.write(f"Error executing code: {str(e)}")


def extract_python_code(response):
    match = re.search(r'```python(.*?)```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


st.markdown("""
    <style>
        .stButton>button {
            background-color: green;
            color: white;
        }
        .stButton>button:hover {
            background-color: darkgreen;
        }
    </style>
""", unsafe_allow_html=True)

user_input = st.text_input("Ask your question:")
if st.button("Generate Answer"):
    if user_input:
        generated_response = chat(user_input)

        generated_code = extract_python_code(generated_response)

        if generated_code:
            with st.expander("Expand to view generated Python code"):

                st.write(f"### Generated Python Code:\n```python\n{generated_code}\n```")

            execute_code(generated_code)

        else:
            st.write("Error: No valid Python code extracted from the response.")
