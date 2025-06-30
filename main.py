"""Main entrypoint with a simple menu for launching subprograms."""

import streamlit as st


def _rerun() -> None:
    """Compatibility wrapper for Streamlit's rerun function."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

import pipe_maze


def show_menu() -> None:
    """Display the start screen with buttons for each activity."""

    st.title("activity books generator")

    if st.button("pipe-maze"):
        st.session_state["page"] = "pipe-maze"
        _rerun()
    if st.button("test2"):
        st.session_state["page"] = "test2"
        _rerun()
    if st.button("test3"):
        st.session_state["page"] = "test3"
        _rerun()
    if st.button("test4"):
        st.session_state["page"] = "test4"
        _rerun()


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
            _rerun()
        pipe_maze.main()
    else:
        st.write(f"{page} clicked")
        if st.button("Back"):
            st.session_state["page"] = "menu"
            _rerun()


if __name__ == "__main__":
    main()


