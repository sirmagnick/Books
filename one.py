"""Streamlit interface embedding the provided Word Search Generator HTML."""

import streamlit as st
from streamlit.components.v1 import html

WORDSEARCH_HTML = """
<!DOCTYPE html>
<html lang="pl">
<head>
  <meta charset="UTF-8">
  <title>Word Search Generator v1.13c</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    h1 { margin-bottom: 10px; }
    .creationContainer { display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 20px; }
    #controls { width: 50%; }
    #gridContainer { flex: 1; max-width: 45%; }
    #grid { display: grid; gap: 1px; margin: 20px 0; width: fit-content; position: relative; }
    .cell { width: 24px; height: 24px; background-color: #fff; border: 1px solid #ccc; text-align: center; vertical-align: middle; line-height: 24px; font-size: 16px; user-select: none; }
    .found { background-color: lightgreen !important; }
    .duplicate { background-color: pink !important; }
    #directions { margin-top: 10px; }
    .direction-row { display: flex; align-items: center; gap: 10px; margin-bottom: 5px; }
    .dir-label { font-size: 20px; cursor: pointer; }
    #randomDirectionContainer { margin-top: 5px; }
    .button { background-color: #4CAF50; color: white; padding: 12px 24px; font-size: 16px; border: none; border-radius: 4px; cursor: pointer; box-shadow: 0px 4px 6px rgba(0,0,0,0.2); margin-right: 5px; }
    .button:hover { background-color: #45a049; }
    #collectionInfo { margin-top: 10px; font-weight: bold; }
    #collectionPreviewFieldset { margin-top: 20px; padding: 10px; border: 1px solid #ccc; width: fit-content; }
    #collectionNav .nav-row { display: flex; align-items: center; margin-bottom: 5px; }
    #collectionIndexInfo { font-weight: bold; margin: 0 10px; }
    #collectionPreview { border: 1px solid #ccc; padding: 10px; width: fit-content; }
    #customSettings { display: none; margin-left: 10px; align-items: center; gap: 10px; }
    #customSettings label { margin-right: 5px; }
    #letterSourceContainer, #puzzleTypeContainer { display: flex; flex-wrap: wrap; gap: 5px; max-width: 100%; }
    #wordCountContainer { display: flex; align-items: center; gap: 5px; }
    #wordCountContainer label { font-size: 16px; }
    #wordCount { width: 60px; }
    #collectionFileInput { display: none; }
    #gridSizeSettingsContainer { margin-top: 10px; display: flex; flex-direction: column; align-items: flex-start; gap: 10px; }
    .gridSettings { display: none; }
    #wordCountDisplay, #collectionWordCountDisplay { margin-bottom: 10px; font-weight: bold; }
    #statusMessage { margin-top: 20px; font-weight: bold; }
  </style>
</head>
<body>

<h1>Word Search Generator v3</h1>

<div class="creationContainer">
  <!-- Lewa kolumna -->
  <div id="controls">
    <div>
      <span id="puzzleTypeContainer">
        <span id="letterSourceContainer">
          <label for="difficultySelect">Poziom trudności:
            <select id="difficultySelect">
              <option value="" disabled selected>wybierz...</option>
              <option value="easy">easy</option>
              <option value="medium">medium</option>
              <option value="hard">hard</option>
              <option value="hell">hell</option>
              <option value="custom">custom</option>
            </select>
          </label>
          <label for="letterSourceSelect">Litery:
            <select id="letterSourceSelect">
              <option value="ze słowa" selected>ze słowa</option>
              <option value="alfabet">alfabet</option>
            </select>
          </label>
          <label for="puzzleTypeSelect">Rodzaj:
            <select id="puzzleTypeSelect">
              <!-- Zostaje tylko multiple -->
              <option value="multiple" selected>multiple</option>
            </select>
          </label>
          <span id="wordCountContainer">
            <label for="wordCount">Liczba słów:</label>
            <input type="number" id="wordCount" value="5" min="1">
          </span>
        </span>
        <span id="customSettings">
          <label>Wiersze:
            <input type="number" id="customRows" value="10" min="1" style="width:60px;">
          </label>
          <label>Kolumny:
            <input type="number" id="customCols" value="8" min="1" style="width:60px;">
          </label>
        </span>
      </span>
    </div>
    <!-- Ustawienia siatki -->
    <div id="gridSizeSettingsContainer">
      <span id="settings_easy" class="gridSettings">
        <label>Wiersze:
          <input type="number" id="easyRows" value="10" min="1" style="width:60px;">
        </label>
        <label>Kolumny:
          <input type="number" id="easyCols" value="8" min="1" style="width:60px;">
        </label>
      </span>
      <span id="settings_medium" class="gridSettings">
        <label>Wiersze:
          <input type="number" id="mediumRows" value="14" min="1" style="width:60px;">
        </label>
        <label>Kolumny:
          <input type="number" id="mediumCols" value="10" min="1" style="width:60px;">
        </label>
      </span>
      <span id="settings_hard" class="gridSettings">
        <label>Wiersze:
          <input type="number" id="hardRows" value="20" min="1" style="width:60px;">
        </label>
        <label>Kolumny:
          <input type="number" id="hardCols" value="14" min="1" style="width:60px;">
        </label>
      </span>
      <span id="settings_hell" class="gridSettings">
        <label>Wiersze:
          <input type="number" id="hellRows" value="32" min="1" style="width:60px;">
        </label>
        <label>Kolumny:
          <input type="number" id="hellCols" value="20" min="1" style="width:60px;">
        </label>
      </span>
      <div>
        <button id="applyGridSizeBtn" class="button">Zatwierdź</button>
        <button id="resetGridSizeBtn" class="button">Resetuj</button>
      </div>
    </div>
    <div style="margin-top:10px;">
      <label>
        Wpisz wyraz (lub wyrazy – po jednym w wierszu):
        <textarea id="wordInput" rows="3" style="width: 100%;">COFFEE</textarea>
      </label>
      <button id="applyWordBtn" class="button">Ustaw wyraz</button>
    </div>
    <div style="margin-top:10px;">
      <label>
        Liczba zadań:
        <input type="number" id="numberOfTasks" value="1" min="1" style="width:60px;">
      </label>
      <button id="addToCollectionBtn" class="button">Do kolekcji</button>
    </div>
    <div style="margin-top:10px;">
      <button id="resetCollectionBtn" class="button">Reset kolekcji</button>
      <button id="exportJsonBtn" class="button">Zapisz na dysk</button>
    </div>
    <div id="collectionInfo">Zapisanych zadań: 0</div>
    <div id="directions" style="margin-top:10px;">
      <div class="direction-row">
        <label class="dir-label"><input type="checkbox" class="dir-chk" data-dr="1" data-dc="0"> ↓</label>
        <label class="dir-label"><input type="checkbox" class="dir-chk" data-dr="0" data-dc="1"> →</label>
        <label class="dir-label"><input type="checkbox" class="dir-chk" data-dr="-1" data-dc="0"> ↑</label>
        <label class="dir-label"><input type="checkbox" class="dir-chk" data-dr="0" data-dc="-1"> ←</label>
      </div>
      <div class="direction-row">
        <label class="dir-label"><input type="checkbox" class="dir-chk" data-dr="1" data-dc="1"> ↘</label>
        <label class="dir-label"><input type="checkbox" class="dir-chk" data-dr="-1" data-dc="1"> ↗</label>
        <label class="dir-label"><input type="checkbox" class="dir-chk" data-dr="1" data-dc="-1"> ↙</label>
        <label class="dir-label"><input type="checkbox" class="dir-chk" data-dr="-1" data-dc="-1"> ↖</label>
      </div>
      <div id="randomDirectionContainer">
        <label class="dir-label"><input type="checkbox" id="dirRandom"> Random</label>
      </div>
      <div id="maxCommonSection" style="margin-top: 10px;">
        <label>Maksymalna liczba wspólnych liter:
          <input type="number" id="maxCommonLetters" value="0" min="0" style="width: 60px;">
        </label>
      </div>
    </div>
  </div>
  <!-- Prawa kolumna -->
  <div id="gridContainer">
    <div id="gridControls" style="margin-bottom:10px;">
      <button id="saveTaskBtn" class="button">Zapisz zadanie</button>
      <button id="newGameBtn" class="button">Nowa gra</button>
      <button id="recalcBtn" class="button">Przelicz</button>
    </div>
    <div id="wordCountDisplay"></div>
    <div id="grid"></div>
    <div id="recalcResult" style="margin-top:10px; font-weight:bold;"></div>
  </div>
</div>
<input type="file" id="collectionFileInput" accept="application/json" />
<fieldset id="collectionPreviewFieldset">
  <legend>Podgląd kolekcji zadań</legend>
  <div id="collectionWordCountDisplay"></div>
  <div id="collectionNav">
    <div class="nav-row">
      <button id="openCollectionBtn" class="button">Otwórz</button>
      <button id="insertTaskBtn" class="button">Wstaw</button>
      <button id="replaceTaskBtn" class="button">Zamień</button>
      <button id="deleteTaskBtn" class="button">Usuń</button>
      <button id="resetCollectionBtn2" class="button">Reset kolekcji</button>
      <button id="exportJsonBtn2" class="button">Zapisz na dysk</button>
    </div>
    <div class="nav-row">
      <button id="firstCollectionBtn" class="button">&lt;&lt;</button>
      <button id="prevCollectionBtn" class="button">Poprzednie zadanie</button>
      <span id="collectionIndexInfo">Zadanie 1 z 0</span>
      <button id="nextCollectionBtn" class="button">Następne zadanie</button>
      <button id="lastCollectionBtn" class="button">&gt;&gt;</button>
    </div>
  </div>
  <div id="collectionPreview">Brak zapisanych zadań</div>
  <div id="duplicatesSummary" style="margin-top:10px; font-weight:bold;"></div>
</fieldset>
<div id="statusMessage" style="margin-top:20px; font-weight:bold;"></div>
<div id="titlesSection" style="margin-top: 10px;">
  <label>
    <input type="checkbox" id="titlesCheckbox">
    tytuły
  </label>
  <br>
  <textarea id="titlesText" rows="4" style="width: 100%;" placeholder="Wpisz tytuły tutaj, każda linia to tytuł dla kolejnego zadania..."></textarea>
</div>
<!-- Tu wklej cały kod JS, bez zmian. Zmieści się jako <script> ... </script> -->
<script>
/* ... KOD JS BEZ ZMIAN (TWÓJ ORYGINALNY ZEWNĘTRZNY SKRYPT) ... */
</script>
</body>
</html>
"""

def main() -> None:
    """Display the embedded wordsearch generator HTML."""
    st.title("Wordsearch One Word")
    html(WORDSEARCH_HTML, height=1000, scrolling=True)

if __name__ == "__main__":
    main()
