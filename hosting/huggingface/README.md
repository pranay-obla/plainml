---
title: plainml
emoji: 📊
colorFrom: red
colorTo: yellow
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

No data to hand? Click one of six example datasets and press Run.

You get headline numbers, plain-English findings, a 3D view you can turn and zoom (every model side by
side, which columns matter, customer groups, unusual rows or forecast lanes) and a full report, and every
result file has its own download button.

**Everything runs in your browser.** Python and plainml run on your own computer (via Pyodide), so your
data is never uploaded anywhere. The first visit downloads about 50 MB of Python libraries; later visits
use the browser's cache. Runs are saved in this browser until you clear its site data.

Made with [plainml](https://github.com/pranay-obla/plainml), which is also at
[plainml-tpua.vercel.app](https://plainml-tpua.vercel.app). To run it on your own machine:
`pip install "plainml[web]"` and `plainml web`.
