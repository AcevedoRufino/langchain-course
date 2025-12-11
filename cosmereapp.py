# Copyright 2025 Snowflake Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from backend.core import run_llm
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
import datetime
import textwrap
import time

import streamlit as st


executor = ThreadPoolExecutor(max_workers=5)
st.set_page_config(page_title="Cosmere Hoid Bot", page_icon="✨")

# -----------------------------------------------------------------------------
# Set things up.

HISTORY_LENGTH = 5
SUMMARIZE_OLD_HISTORY = True
DOCSTRINGS_CONTEXT_LEN = 10
PAGES_CONTEXT_LEN = 10
MIN_TIME_BETWEEN_REQUESTS = datetime.timedelta(seconds=3)

DEBUG_MODE = st.query_params.get("debug", "false").lower() == "true"

INSTRUCTIONS = textwrap.dedent("""
    - You are a helpful AI chat assistant focused on answering quesions about
      Brandon Sanderson's Cosmere Universe.
    - You will be given context information and should only answer from that context.
    - If the context does not contain the answer, specifically say "I don't know".
    - If the question is not about the Cosmere Universe, say "Not in the Cosmere"
    - Use context and history to provide a coherent answer.
    - Use markdown such as headers (starting with ##), bullet
      points, indentation for sub bullets
    - Don't start the response with a markdown header.
    - Be brief, but clear. If needed, you can write paragraphs of text, like
      a documentation website.
    - Avoid experimental and private APIs.
    - Use a friendly and warm tone.
    
""")

SUGGESTIONS = {
    ":blue[:material/local_library:] What is a Spren?": (
        "What is a Spren, what is it great at, and who can bond with it?"
    ),
    ":green[:material/database:] Help me understand Shards": (
        "Help me understand Shards. How many are there? "
        "What entities make up each Shard?"
    ),
    ":orange[:material/multiline_chart:] Who is Hoid?": (
        "Who is Hoid? What role does he play in the Cosmere Universe?"
    ),
    ":violet[:material/wand_stars:] Magic Systems?": (
        "What are the different magic systems in the Cosmere Universe? How do they work?"
    ),
    ":red[:material/bloodtype:] Ghostbloods": (
        "Who are the Ghostbloods? What role do they play in the Cosmere Universe?"
    ),
}

def create_sources_string(source_urls: set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


def build_prompt(**kwargs):
    """Builds a prompt string with the kwargs as HTML-like tags.

    For example, this:

        build_prompt(foo="1\n2\n3", bar="4\n5\n6")

    ...returns:

        '''
        <foo>
        1
        2
        3
        </foo>
        <bar>
        4
        5
        6
        </bar>
        '''
    """
    prompt = []

    for name, contents in kwargs.items():
        if contents:
            prompt.append(f"<{name}>\n{contents}\n</{name}>")

    prompt_str = "\n".join(prompt)

    return prompt_str


# Just some little objects to make tasks more readable.
TaskInfo = namedtuple("TaskInfo", ["name", "function", "args"])
TaskResult = namedtuple("TaskResult", ["name", "result"])


def build_question_prompt(question):
    """Fetches info from different services and creates the prompt string."""
    old_history = st.session_state.messages[:-HISTORY_LENGTH]
    recent_history = st.session_state.messages[-HISTORY_LENGTH:]

    if recent_history:
        recent_history_str = history_to_text(recent_history)
    else:
        recent_history_str = None

    # Fetch information from different services in parallel.
    task_infos = []

    if SUMMARIZE_OLD_HISTORY and old_history:
        task_infos.append(
            TaskInfo(
                name="old_message_summary",
                function=generate_chat_summary,
                args=(old_history,),
            )
        )

    results = executor.map(
        lambda task_info: TaskResult(
            name=task_info.name,
            result=task_info.function(*task_info.args),
        ),
        task_infos,
    )

    context = {name: result for name, result in results}

    return build_prompt(
        instructions=INSTRUCTIONS,
        **context,
        recent_messages=recent_history_str,
        question=question,
    )


def history_to_text(chat_history):
    """Converts chat history into a string."""
    return "\n".join(f"[{h['role']}]: {h['content']}" for h in chat_history)


@st.dialog("Disclaimer")
def show_disclaimer_dialog():
    st.caption("""
            I am an AI spren bot tasked by Hoid to provide you all 
            information. Answers may be inaccurate, inefficient, or biased.
            Any use or decisions based on such answers should include reasonable
            practices including human oversight to ensure they are safe,
            accurate, and suitable for your intended purpose. Hoid is not
            liable for any actions, losses, or damages resulting from the use
            of this bot. Do not enter any private, sensitive, personal, or
            regulated data. We do not have permission from Brandon Sanderson,
            Dragon Steel, or his publishers to use this material. For more
            official information on the Cosmere, see
            https://www.brandonsanderson.com/.
        """)


# -----------------------------------------------------------------------------
# Draw the UI.


#st.html(div(style=styles(font_size=rem(5), line_height=1))["❉"])
#st.html(div(style=styles(font_size=rem(5), line_height=1))[st.image('dragonsteel.webp')])
col1, col2 = st.columns([1, 4])  # Adjust ratio as needed

with col1:
    st.image("dragonsteel.webp", width=150)

with col2:
    title_row = st.container(
        horizontal=True,
        vertical_alignment="bottom",
    )
    with title_row:
        st.markdown('<h1 style="color:#770B12;">Cosmere Hoid Bot ✨</h1>', unsafe_allow_html=True)

    user_just_asked_initial_question = (
        "initial_question" in st.session_state and st.session_state.initial_question
    )

    user_just_clicked_suggestion = (
        "selected_suggestion" in st.session_state and st.session_state.selected_suggestion
    )

    user_first_interaction = (
        user_just_asked_initial_question or user_just_clicked_suggestion
    )

    has_message_history = (
        "messages" in st.session_state and len(st.session_state.messages) > 0
    )

    # Show a different UI when the user hasn't asked a question yet.
    if not user_first_interaction and not has_message_history:
        st.session_state.messages = []

        with st.container():
            st.chat_input("Ask a question about the Cosmere...", key="initial_question")

            selected_suggestion = st.pills(
                label="Examples",
                label_visibility="collapsed",
                options=SUGGESTIONS.keys(),
                key="selected_suggestion",
            )

        st.button(
            "&nbsp;:small[:gray[:material/balance: Disclaimer]]",
            type="tertiary",
            on_click=show_disclaimer_dialog,
        )

        st.stop()

    # Show chat input at the bottom when a question has been asked.
    with st._bottom:
        bottom_col1, bottom_col2 = st.columns([1, 4])
        with bottom_col1:
            def clear_conversation():
                st.session_state.messages = []
                st.session_state.initial_question = None
                st.session_state.selected_suggestion = None

            st.button(
                "New Chat",
                icon=":material/refresh:",
                on_click=clear_conversation,
            )
        with bottom_col2:
            user_message = st.chat_input("Ask a follow-up...")

    if not user_message:
        if user_just_asked_initial_question:
            user_message = st.session_state.initial_question
        if user_just_clicked_suggestion:
            user_message = SUGGESTIONS[st.session_state.selected_suggestion]


    if "prev_question_timestamp" not in st.session_state:
        st.session_state.prev_question_timestamp = datetime.datetime.fromtimestamp(0)

    # Display chat messages from history as speech bubbles.
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.container()  # Fix ghost message bug.

            st.markdown(message["content"])

   
    if user_message:
        # When the user posts a message...

        # Streamlit's Markdown engine interprets "$" as LaTeX code (used to
        # display math). The line below fixes it.
        user_message = user_message.replace("$", r"\$")

        # Display message as a speech bubble.
        with st.chat_message("user"):
            st.text(user_message)

        # Display assistant response as a speech bubble.
        with st.chat_message("assistant"):
            with st.spinner("Waiting..."):
                # Rate-limit the input if needed.
                question_timestamp = datetime.datetime.now()
                time_diff = question_timestamp - st.session_state.prev_question_timestamp
                st.session_state.prev_question_timestamp = question_timestamp

                if time_diff < MIN_TIME_BETWEEN_REQUESTS:
                    time.sleep(time_diff.seconds + time_diff.microseconds * 0.001)

                user_message = user_message.replace("'", "")

            # Build a detailed prompt.
            if DEBUG_MODE:
                with st.status("Computing prompt...") as status:
                    full_prompt = build_question_prompt(user_message)
                    st.code(full_prompt)
                    status.update(label="Prompt computed")
            else:
                with st.spinner("Researching..."):
                    full_prompt = build_question_prompt(user_message)

            # Send prompt to LLM.
            with st.spinner("Thinking..."):
                generated_response = run_llm(
                    query=full_prompt, chat_history=st.session_state.messages
                )
                sources = set(
                    [doc.metadata["source"] for doc in generated_response["source_documents"]]
                )

                if str(generated_response['result']).strip() == "Not in the Cosmere" or str(generated_response['result']).strip() == "I don't know":
                    formatted_response = f"{generated_response['result']}"
                else:
                    formatted_response = (
                        f"{generated_response['result']} \n\n {create_sources_string(sources)}"
                    )
            # Put everything after the spinners in a container to fix the
            # ghost message bug.
            with st.container():

                response = st.write(formatted_response)
                # Add messages to chat history.
                st.session_state.messages.append({"role": "user", "content": user_message})
                st.session_state.messages.append({"role": "assistant", "content": formatted_response})


            