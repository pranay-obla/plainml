"use client";

// The before and after: churn.csv as it arrives, and what plainml wrote about it (the real run's words).

import { useEffect, useState } from "react";

const COLUMNS = ["customer_id", "plan", "contract", "monthly_charges", "support_calls", "churned"];
// The first five rows of plainml/datasets/churn.csv, these columns only.
const ROWS = [
  ["C100000", "standard", "monthly", "$68.50", "1", "no"],
  ["C100001", "basic", "annual", "$23.29", "0", "no"],
  ["C100002", "premium", "monthly", "$103.50", "0", "no"],
  ["C100003", "basic", "annual", "$23.87", "2", "no"],
  ["C100004", "basic", "monthly", "$28.17", "2", "yes"],
];
// From the run's own findings (20261124-205857_churn).
const FINDINGS = [
  "The model relies most on signup_date, support_calls and contract.",
  "More support calls, more likely to leave: 24% rising to 38%.",
  "Monthly contracts leave most (37%), annual ones least (18%).",
  "Premium plans leave most (39%), basic ones least (22%).",
];

export function BeforeAfter() {
  const [after, setAfter] = useState(true);
  // without a script both panes stand in the flow; with one, they share a place and the switch picks
  const [live, setLive] = useState(false);
  useEffect(() => setLive(true), []);
  return (
    <div className={live ? "ba is-live" : "ba"}>
      <div className="switch" role="group" aria-label="Before or after plainml" hidden={!live}>
        <button type="button" aria-pressed={!after} onClick={() => setAfter(false)}>
          Your spreadsheet
        </button>
        <button type="button" aria-pressed={after} onClick={() => setAfter(true)}>
          What plainml tells you
        </button>
      </div>
      <div className="ba__panes">
        <div className="ba__pane ba__before" data-on={!after} aria-hidden={live && after}>
          <p className="ba__file">churn.csv, 1,500 rows</p>
          <div className="ba__scroll">
            <table className="sheet">
              <thead>
                <tr>
                  {COLUMNS.map((c) => (
                    <th key={c} scope="col">
                      {c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {ROWS.map((row) => (
                  <tr key={row[0]}>
                    {row.map((cell, i) => (
                      <td key={i} className={i === 3 || i === 4 ? "num" : undefined}>
                        {cell}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div className="ba__pane ba__after" data-on={after} aria-hidden={live && !after}>
          <p className="ba__file">Who is likely to leave, from churn.csv</p>
          <p className="ba__score">
            It scores <b className="num">0.64</b> out of 1 on customers it never saw. A simple guess scores <b className="num">0.45</b>.
          </p>
          <ul className="ba__findings">
            {FINDINGS.map((f) => (
              <li key={f}>{f}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
