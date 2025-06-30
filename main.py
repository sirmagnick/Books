import streamlit as st
import pipe_maze

st.title("activity books generator")

if st.button("pipe-maze"):
    pipe_maze.main()
if st.button("test2"):
    st.write("Test2 clicked")
if st.button("test3"):
    st.write("Test3 clicked")
if st.button("test4"):
    st.write("Test4 clicked")

