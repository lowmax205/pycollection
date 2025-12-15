# URL Scraper (Tkinter) — Presentation Script

## 1) Opening (15–30s)
**What I’ll say**
- “Today I’m presenting a small desktop app written in Python using Tkinter.”
- “It takes a website URL, downloads the page, parses the HTML, and extracts a few useful things: the page title, headings, emails found in the raw text, and a list of external links.”
- “It can also save the scraped results into a CSV file.”

**What to show on screen**
- Open the app window.
- Point at the URL input, the output area, and the Save/Clear buttons.

---

## 2) What this program does (30–45s)
**Key features**
- URL input with Enter-to-scrape.
- Scrapes a page using `requests` (HTTP) and `BeautifulSoup` (HTML parsing).
- Extracts:
  - Title (`<title>...</title>`)
  - Headings (`<h1>`, `<h2>`, `<h3>`) (limited)
  - Emails (regex from the response text)
  - Links (unique, external `http...` links, limited)
- Displays results inside a scrollable text area.
- Saves results to a timestamped CSV.

**Why it’s useful**
- “It’s a simple example of combining a GUI + web requests + HTML parsing + exporting data.”

---

## 3) Dependencies and setup (30–45s)
**What I’ll say**
- “Tkinter ships with most Python installs, so the GUI usually works immediately.”
- “For web scraping, we use two external libraries: `requests` and `beautifulsoup4`.”

**Commands to mention**
- Install dependencies:
  - `pip install requests beautifulsoup4`

**How the code handles missing packages**
- The script tries importing these libraries in a `try/except ImportError` block.
- If they’re missing, the app shows a message box telling the user what to install.

---

## 4) High-level architecture (45–60s)
**What I’ll say**
- “This is a single-file Tkinter app.”
- “There are three main actions, each connected to a button or event:”
  1. `scrape_website()` — download and extract data, then render it.
  2. `save_data()` — write the stored results to a CSV file.
  3. `clear_all()` — reset the UI for a fresh run.

**Data flow**
- UI input (`url_entry`) → `scrape_website()`
- Network response (`requests.get`) → parse (`BeautifulSoup`)
- Extracted values → displayed in `output_text` and stored in `scraped_data`
- `save_data()` reads from `scraped_data` → creates `scraped_data_YYYYMMDD_HHMMSS.csv`

---

## 5) Walkthrough: `scrape_website()` (2–3 minutes)
**What I’ll say**
- “First, it reads the URL from the Entry widget and does basic validation.”
- “If the URL doesn’t start with `http://` or `https://`, it adds `https://` by default.”
- “Before starting the request, it updates the output area to ‘Scraping…’ and disables the Scrape button to prevent double-click spam.”

**Network step**
- Uses a `User-Agent` header (some sites block requests that look like bots).
- Uses `timeout=10` and `raise_for_status()`.

**Parsing and extraction**
- Title:
  - `soup.title.string.strip()` if `<title>` exists.
- Links:
  - Find all `<a href=...>`.
  - Only keep links starting with `http`.
  - Remove duplicates with `set`, then limit to 20.
- Emails:
  - Regex search on `response.text`.
  - Stored in a set to avoid duplicates.
- Headings:
  - Collect `<h1>`, `<h2>`, `<h3>` text.
  - Limit to 10.

**Output**
- Writes a clear report into the scrolled text box:
  - Title
  - Heading list
  - Emails (or “No emails found”)
  - Links

**Storage**
- Saves results in a dictionary called `scraped_data` with keys:
  - `url`, `title`, `headings`, `emails`, `links`, `timestamp`

---

## 6) Error handling (1 minute)
**What I’ll say**
- “Web requests can fail for lots of reasons: DNS issues, timeouts, blocked requests, or invalid URLs.”
- The code handles:
  - `requests.exceptions.RequestException`: shows a friendly error plus tips.
  - Generic `Exception`: catches anything unexpected and shows the message.
- The `finally` block re-enables the Scrape button.

**Good practice point**
- “Disabling/re-enabling buttons around long actions is a simple UX improvement.”

---

## 7) Walkthrough: `save_data()` (45–60s)
**What I’ll say**
- “Saving only works after a successful scrape. The Save button starts disabled and is enabled after scraping.”
- It saves a CSV file with a timestamped name like:
  - `scraped_data_20251215_143200.csv`
- The CSV format is simple:
  - Some metadata rows (URL, Title, Timestamp)
  - Then sections for Headings, Emails, Links

**Why CSV**
- “CSV is easy to open in Excel/Google Sheets and easy to process later.”

---

## 8) Walkthrough: `clear_all()` (15–30s)
**What I’ll say**
- “Clear resets the URL box, clears the output, disables Save again, and resets status text.”
- “It also clears `scraped_data`, so you can’t accidentally save old data.”

---

## 9) Quick demo plan (1–2 minutes)
**Step-by-step demo**
1. Enter a safe, public site (examples):
   - `python.org`
   - `wikipedia.org`
2. Click Scrape (or press Enter).
3. Point out:
   - Title
   - Heading list
   - Emails section (often empty)
   - Links section
4. Click Save to CSV and show the saved filename.
5. Click Clear to reset.

---

## 10) Limitations and improvements (optional, 45–90s)
**Honest limitations**
- Some sites block scraping or require JavaScript rendering.
- Email regex can find false positives or miss obfuscated emails.
- Link extraction currently keeps only `http...` links; it ignores relative links.
- The UI runs the request in the main thread, so the window may feel “busy” during slow requests.

**Possible improvements**
- Run network scraping in a background thread.
- Support relative links via `urllib.parse.urljoin(base_url, href)`.
- Export JSON as well as CSV.
- Add a progress indicator/spinner.

---

## 11) Closing (15–30s)
**What I’ll say**
- “This project demonstrates a full pipeline: GUI input → HTTP fetch → HTML parsing → data extraction → user-friendly output → file export.”
- “The code stays beginner-friendly while still using real libraries used in production.”

---

## Appendix: “Where to look in the code” (for Q&A)
- Imports and dependency check: top of file
- Scraping logic: `scrape_website()`
- Export: `save_data()`
- Reset: `clear_all()`
- Tkinter UI building: bottom half of the file
