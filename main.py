"""Main entrypoint with a simple menu for launching subprograms."""

import streamlit as st

import pipe_maze


def show_menu() -> None:
    """Display the start screen with buttons for each activity."""

    st.title("activity books generator")

    if st.button("pipe-maze"):
        st.session_state["page"] = "pipe-maze"
        st.experimental_rerun()
    if st.button("test2"):
        st.session_state["page"] = "test2"
        st.experimental_rerun()
    if st.button("test3"):
        st.session_state["page"] = "test3"
        st.experimental_rerun()
    if st.button("test4"):
        st.session_state["page"] = "test4"
        st.experimental_rerun()


def main() -> None:
    """Route between the menu and subprograms using session state."""

    if "page" not in st.session_state:
        st.session_state["page"] = "menu"

    page = st.session_state["page"]

    if page == "menu":
        show_menu()
    elif page == "pipe-maze":
        if st.button("Back"):
            st.session_state["page"] = "menu"
            st.experimental_rerun()
        pipe_maze.main()
    else:
        st.write(f"{page} clicked")
        if st.button("Back"):
            st.session_state["page"] = "menu"
            st.experimental_rerun()


if __name__ == "__main__":
    main()


