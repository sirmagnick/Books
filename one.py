"""Streamlit wrapper to run a simple one-word wordsearch game."""

import streamlit as st
from streamlit.components.v1 import html

WORDSEARCH_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Wordsearch One Word</title>
<style>
  body { font-family: sans-serif; }
  #grid { border-collapse: collapse; margin-top: 1em; }
  #grid td { width: 40px; height: 40px; border: 1px solid #999; text-align: center;
             font-size: 20px; cursor: pointer; }
  .found { background: lightgreen; }
</style>
</head>
<body>
<h3>Find the word: PYTHON</h3>
<table id="grid"></table>
<script>
const word = "PYTHON";
const size = 10;
const grid = [];
for (let r = 0; r < size; r++) {
  grid[r] = [];
  for (let c = 0; c < size; c++) {
    grid[r][c] = String.fromCharCode(65 + Math.floor(Math.random() * 26));
  }
}
const row = Math.floor(Math.random() * size);
const col = Math.floor(Math.random() * (size - word.length));
for (let i = 0; i < word.length; i++) {
  grid[row][col + i] = word[i];
}
const tbl = document.getElementById('grid');
for (let r = 0; r < size; r++) {
  const tr = document.createElement('tr');
  for (let c = 0; c < size; c++) {
    const td = document.createElement('td');
    td.textContent = grid[r][c];
    tr.appendChild(td);
  }
  tbl.appendChild(tr);
}
let selection = [];
tbl.addEventListener('click', e => {
  if (e.target.tagName === 'TD') {
    const td = e.target;
    td.classList.add('found');
    selection.push(td);
    if (selection.length === word.length) {
      let match = true;
      for (let i = 0; i < selection.length; i++) {
        if (selection[i].textContent !== word[i]) {
          match = false;
          break;
        }
      }
      if (match) {
        alert('Congratulations! You found the word!');
      } else {
        alert('Incorrect selection, try again.');
      }
      selection.forEach(cell => cell.classList.remove('found'));
      selection = [];
    }
  }
});
</script>
</body>
</html>
"""

def main() -> None:
    """Display the embedded wordsearch game using Streamlit."""
    st.title("Wordsearch One Word")
    html(WORDSEARCH_HTML, height=600)

if __name__ == "__main__":
    main()
