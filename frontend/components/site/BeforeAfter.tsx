"use client";

// The before and after: churn.csv as it arrives, and what plainml wrote about it (the real run's words).

import { useEffect, useRef, useState } from "react";

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
  "This model leans most on signup_date, support_calls and contract.",
  "More support calls, more likely to leave: 24% rising to 38%.",
  "Monthly contracts leave most (37%), annual ones least (18%).",
  "Premium plans leave most (39%), basic ones least (22%).",
];

export function BeforeAfter() {
  const [after, setAfter] = useState(false);
  // without a script both panes stand in the flow; with one, they share a place and the switch picks
  const [live, setLive] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const touched = useRef(false);
  useEffect(() => {
    setLive(true);
    // it opens on the spreadsheet and turns into the findings once it is on screen, unless the
    // visitor has already picked a side; with reduced motion it opens on the findings
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setAfter(true);
      return;
    }
    const el = ref.current;
    if (!el) return;
    let timer = 0;
    const io = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) return;
        io.disconnect();
        timer = window.setTimeout(() => {
          if (!touched.current) setAfter(true);
        }, 1600);
      },
      { threshold: 0.6 },
    );
    io.observe(el);
    return () => {
      io.disconnect();
      window.clearTimeout(timer);
    };
  }, []);
  const pick = (value: boolean) => {
    touched.current = true;
    setAfter(value);
  };
  return (
    <div className={live ? "ba is-live" : "ba"} ref={ref}>
      <div className="switch" role="group" aria-label="Before or after plainml" hidden={!live}>
        <button type="button" aria-pressed={!after} onClick={() => pick(false)}>
          Your spreadsheet
        </button>
        <button type="button" aria-pressed={after} onClick={() => pick(true)}>
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
            On customers it never saw it scores <b className="num">0.64</b> out of 1, where higher is better. A simple guess scores <b className="num">0.45</b>.
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
