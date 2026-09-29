---
title: plainml
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
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

**Tip:** open the app on its own page, at `https://<owner>-<space-name>.hf.space`, instead of inside the
Hugging Face page. Uploads and sign-in work best there.

Runs made here are wiped when the Space restarts, unless it has persistent storage. Download what you
want to keep.

Made with [plainml](https://github.com/pranay-obla/plainml). To run it on your own machine:
`pip install "plainml[web]"` and `plainml web`.
