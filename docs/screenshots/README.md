# Application Screenshots Guide

Screenshots of the running Orient BlackSwan History Bot Streamlit application (`http://localhost:8501`) can be saved here for the 5 assignment benchmark questions:

| Filename | Question / Screen | Expected Result |
| :--- | :--- | :--- |
| `q1_price_update_date.png` | *On what date were the prices in this catalogue last updated?* | Supported answer with citation Page `ii`, `iii`, `1` |
| `q2_incorporation_year.png` | *In which year was the publisher (Orient BlackSwan) incorporated?* | Supported answer with citation Page `ii` (1948) |
| `q3_physics_textbook.png` | *What is the price of a Physics textbook?* | Strict `Not found` fallback |
| `q4_education_minister.png` | *Who is the current Minister of Education?* | Strict `Not found` fallback |
| `q5_beginner_history_book.png` | *Which is the 'best' history book for a beginner?* | Strict `Not found` fallback |

### Capturing Genuine Screenshots on macOS:
1. Ensure the app is running: `streamlit run app.py`
2. Open `http://localhost:8501` in Safari or Chrome.
3. For each question, type the query, click **Ask Question**, and press **`Cmd + Shift + 4`** to capture the browser window.
4. Save the resulting screenshots in `docs/screenshots/` with the corresponding filenames above.

