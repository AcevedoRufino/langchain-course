from backend.core import run_llm
from dotenv import load_dotenv
import streamlit as st

load_dotenv()


def render_user_sidebar() -> None:
    """Render the sidebar section with basic user info."""
    with st.sidebar:
        st.header("User")
        streamlit_user = getattr(st, "experimental_user", None)

        display_name = "Guest"
        email = "Not signed in"
        avatar_url = "https://static.streamlit.io/examples/dice.jpg"

        if streamlit_user and getattr(streamlit_user, "email", None):
            email = streamlit_user.email
            display_name = getattr(streamlit_user, "name", email.split("@")[0])
            avatar_url = getattr(streamlit_user, "avatar_url", avatar_url)

        st.image(avatar_url, width=96)
        st.markdown(f"**Name:** {display_name}")
        st.markdown(f"**Email:** {email}")

def create_sources_string(source_urls: set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

def main():
    st.header("Hoid Spren Bot")
    render_user_sidebar()
    prompt = st.text_input("Prompt", placeholder="What can I help with?")

    if (
        "chat_answer_history" not in st.session_state
        and "user_prompt_history" not in st.session_state
        and "chat_history" not in st.session_state
    ):
        st.session_state["chat_answer_history"] = []
        st.session_state["user_prompt_history"] = []
        st.session_state["chat_history"] = []

    if prompt:
        with st.spinner("Thinking..."):
            generated_response = run_llm(
                query=prompt, chat_history=st.session_state["chat_history"]
            )
            sources = set(
                [doc.metadata["source"] for doc in generated_response["source_documents"]]
            )

            formatted_response = (
                f"{generated_response['result']} \n\n {create_sources_string(sources)}"
            )
            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_answer_history"].append(formatted_response)
            st.session_state["chat_history"].append(("human", prompt))
            st.session_state["chat_history"].append(("ai", generated_response["result"]))


    if st.session_state["chat_answer_history"]:
        for generated_response, user_query in zip(st.session_state["chat_answer_history"], st.session_state["user_prompt_history"]):
            st.chat_message("user").write(user_query)
            st.chat_message("assistant").write(generated_response)

if __name__ == "__main__":
    main()
