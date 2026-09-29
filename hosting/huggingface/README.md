---
title: plainml
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Upload a spreadsheet, get an explained ML model
---

# plainml

Upload a CSV or Excel file and choose what to find out:
- predict a column
- forecast over time
- find groups or unusual rows
- rank the columns
- check for drift
- profile or clean the data

You get headline numbers, plain-English findings and a full report, and every result file has its own
download button.

**Everything runs in your browser.** Python and plainml run on your own computer (via Pyodide), so your
data is never uploaded anywhere. The first visit downloads about 50 MB of Python libraries; later visits
use the browser's cache. Runs are saved in this browser until you clear its site data.

Made with [plainml](https://github.com/pranay-obla/plainml). To run it on your own machine:
`pip install "plainml[web]"` and `plainml web`.
