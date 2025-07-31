"""Main entrypoint with a simple menu for launching subprograms."""

import streamlit as st


def _rerun() -> None:
    """Compatibility wrapper for Streamlit's rerun function."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

import importlib.util


def _run_module(path: str) -> None:
    """Load and execute a module's main() function from a path."""
    spec = importlib.util.spec_from_file_location(path, path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "main"):
            module.main()


def run_3dmaze() -> None:
    """Launch the 3D maze program."""
    _run_module("3dmaze.py")


def run_multi_maze() -> None:
    """Launch the multi-maze program."""
    _run_module("multi_maze.py")


def show_menu() -> None:
    """Display the start screen with buttons for each activity."""

    st.title("activity books generator")

    if st.button("3d maze"):
        st.session_state["page"] = "3d maze"
        _rerun()
    if st.button("multi-maze"):
        st.session_state["page"] = "multi-maze"
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
    elif page == "3d maze":
        if st.button("Back"):
            st.session_state["page"] = "menu"
            _rerun()
        run_3dmaze()
    elif page == "multi-maze":
        if st.button("Back"):
            st.session_state["page"] = "menu"
            _rerun()
        run_multi_maze()
    else:
        st.write(f"{page} clicked")
        if st.button("Back"):
            st.session_state["page"] = "menu"
            _rerun()


if __name__ == "__main__":
    main()


