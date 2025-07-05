import streamlit as st
import asyncio
import sys
from main import handle_query
from query_classifier import is_valid_query

# To handle event loop in non terminal os like window
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

st.set_page_config(page_title="Web QA Agent", layout="centered")
st.title("🔎 Ask Anything from the Web")

user_query = st.text_input("Enter your question:")
result_placeholder = st.empty()
if st.button("Search"):
    if is_valid_query(user_query) and len(user_query) > 15:
        with st.spinner("Searching and summarizing..."):
            loop = asyncio.get_event_loop()
            answer = loop.run_until_complete(handle_query(user_query))
            result_placeholder.markdown(f"**Answer:**\n\n{answer}")
    else:
        st.warning("Please enter a valid question.")
