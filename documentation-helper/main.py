from typing import Set

from backend.core import run_llm
import streamlit as st
from streamlit_chat import message

# Custom CSS for profile styling and background
st.markdown("""
<style>
/* Global background style */
.stApp {
    background-color: #FFF6E6;  /* 파스텔 아이보리 배경 */
}

.profile-container {
    background-color: #FFFFFF;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.profile-image {
    width: 150px;
    height: 150px;
    border-radius: 50%;
    margin: 0 auto 15px auto;
    display: block;
    object-fit: cover;
    border: 3px solid #FFF6E6;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.profile-name {
    font-size: 1.5em;
    font-weight: bold;
    text-align: center;
    margin-bottom: 5px;
    color: #4A4A4A;
}

.profile-email {
    color: #777;
    text-align: center;
    font-size: 0.9em;
    margin-bottom: 15px;
}

.profile-divider {
    margin: 10px 0;
    border-top: 1px solid #EEE;
}

/* Chat message container styling */
.stChatMessage {
    background-color: #FFFFFF !important;
    border-radius: 15px;
    padding: 10px 15px;
    margin: 5px 0;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

/* Header styling */
.stHeader {
    background-color: transparent !important;
}

/* Input field styling */
.stTextInput > div > div {
    background-color: #FFFFFF;
    border-radius: 10px;
    border: 1px solid #EEE;
}

</style>
""", unsafe_allow_html=True)

# Sidebar with profile
with st.sidebar:
    st.markdown("""
    <div class="profile-container">
        <img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/user.svg" class="profile-image">
        <div class="profile-name">John Doe</div>
        <div class="profile-email">john.doe@example.com</div>
        <div class="profile-divider"></div>
    </div>
    """, unsafe_allow_html=True)

st.header("LangChain Udemcleary Course- Documentation Helper Bot")


prompt = st.text_input("Prompt", placeholder="Enter your prompt here....")

if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []


def create_source_string(sources_urls: Set[str]) ->str:
    if not sources_urls:
        return ""
    sources_list = List(sources_urls)
    sources_list.sort()
    sources_string = "source\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


if prompt:
    with st.spinner("Generating response.."):
        # import time
        # time.sleep(3)
        generated_response = run_llm(
            query=prompt, chat_history=st.session_state["chat_history"]
        )
        sources = set(
            [doc.metadata["source"] for doc in generated_response["source_documents"]]
        )

        formatted_response = (
            f"{generated_response['result']} \n\n {create_source_string(sources)}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["result"]))


if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
            st.session_state["chat_answers_history"],
            st.session_state["user_prompt_history"]
    ):
        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(generated_response)
