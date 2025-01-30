import os
import streamlit as st
from streamlit_chat import message

from add_review import add_review
# from lib.utils import get_docs_and_headers
from llm import LLM

os.environ['GROQ_API_KEY'] = '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'


def build_chat_ui(docs: list[str], chunk_header_map: dict[str, str]):
    if 'drug_name' not in st.session_state:
        st.session_state['drug_name'] = ''
    if 'user_review' not in st.session_state:
        st.session_state['user_review'] = ''
    if 'add_review_triggered' not in st.session_state:
        st.session_state['add_review_triggered'] = False

    with st.sidebar:
        st.session_state['drug_name'] = st.text_input("Drug name", value=st.session_state['drug_name'])
        st.session_state['user_review'] = st.text_input("Write review here", value=st.session_state['user_review'])
        add_review_btn = st.button("Add review")

    if add_review_btn:
        st.session_state['add_review_triggered'] = True

    if st.session_state['add_review_triggered']:
        if st.session_state['drug_name'] and st.session_state['user_review']:
            add_review(st.session_state['drug_name'], st.session_state['user_review'])

            st.session_state['drug_name'] = ''
            st.session_state['user_review'] = ''
            st.session_state['add_review_triggered'] = False
            # st.experimental_rerun()
        else:
            with st.sidebar:
                st.warning("Fill in all fields")
            st.session_state['add_review_triggered'] = False

    if "messages" not in st.session_state:
        greeting = "Hi there :)"
        st.session_state.messages = [{"role": "assistant", "content": greeting}]

    for i, m in enumerate(st.session_state.messages):
        if m["role"] == "user":
            message(m["content"], is_user=True, key=f"user_{i}")
        else:
            message(m["content"], key=f"assistant_{i}")
            if "sources" in m:
                st.markdown("### Sources:")
                cols = st.columns(len(m["sources"]))
                for idx, cont_i in enumerate(m["sources"]):
                    with cols[idx]:
                        st.caption(f"Closest subtopic:\n {cont_i['closest_subtopic']}"
                                   f"\nContent:\n {cont_i['content_snippet']}...")
                        st.markdown(f"Link: {cont_i['link']}")

    llm = LLM(docs, chunk_header_map)
    query = st.chat_input("Say something")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        message(query, is_user=True,
                key=f"user_input_{len(st.session_state.messages)}")

        answer, context, headers = llm.answer_question(query)

        message(answer, key=f"assistant_response_{len(st.session_state.messages)}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })


if __name__ == "__main__":
    all_docs, all_chunk_header_map = [], {}  # get_docs_and_headers()
    build_chat_ui(all_docs, all_chunk_header_map)
