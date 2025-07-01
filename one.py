import streamlit as st
from streamlit.components.v1 import html

HTML_FILE = "wordsearch.html"


def main() -> None:
    """Render the word search generator HTML using Streamlit."""
    with open(HTML_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    st.title("Wordsearch One Word")
    html(content, height=1000, scrolling=True)


if __name__ == "__main__":
    main()
