import type { Metadata } from "next";
import type { Visual } from "@/lib/types";
import visuals from "@/public/data/visuals.json";
import { BeforeAfter } from "@/components/site/BeforeAfter";
import { CopyCommand } from "@/components/site/CopyCommand";
import { HeldOut } from "@/components/site/HeldOut";
import { Jobs } from "@/components/site/Jobs";
import { SceneStage } from "@/components/site/SceneStage";
import "./site.css";

// plainml's home page: a personal project page by Pranay Obla. Every number on it comes from a real run
// on plainml's example datasets (public/data/visuals.json and the run 20261124-205857_churn).
// Links to the app are plain <a> (a full page load), so this page's styles never reach the app.

export const metadata: Metadata = {
  title: "plainml: machine learning without learning machine learning",
  description:
    "Drop in a spreadsheet and get the best model, in plain English. Free and open source; Python runs in your browser, so your data stays on your computer.",
};

const GITHUB = "https://github.com/pranay-obla/plainml";
const ME = "https://github.com/pranay-obla";

const models = visuals.models as unknown as Visual;
const best = (models.type === "bars" && (models.bars.find((b) => b.tone === "best") || models.bars[0])) || null;
const baseline = models.type === "bars" ? models.lines?.[0]?.value : undefined;
// the run's test score: the winner on the 300 customers it never saw (run 20261124-205857_churn, "Test F1")
const TEST = 0.6358;
// the hero shows the simple guess as a bar of its own, so the gap to it is the picture
const heroVisual: Visual =
  models.type === "bars" && baseline != null
    ? { ...models, value_label: "Score", lines: [], bars: [...models.bars, { label: "A simple guess", value: baseline, tone: "bad" }] }
    : models;

function Arrow() {
  return (
    <svg className="arrow" viewBox="0 0 16 16" aria-hidden="true">
      <path d="M3 8h9M8.5 4.5 12 8l-3.5 3.5" />
    </svg>
  );
}

export default function Home() {
  return (
    <div className="site">
      <a className="skip" href="#main">
        Skip to content
      </a>
      <header className="band">
        <nav className="nav container" aria-label="Main">
          <a className="nav__mark" href="/">
            <img src="/favicon.svg" alt="" width="28" height="28" />
            <span>plainml</span>
          </a>
          <a className="nav__by" href={ME}>
            by Pranay Obla
          </a>
          <ul className="nav__links">
            <li>
              <a href="#tested">How it&apos;s tested</a>
            </li>
            <li>
              <a href="#jobs">What it does</a>
            </li>
            <li>
              <a href="#where">Where it runs</a>
            </li>
            <li>
              <a href={GITHUB}>GitHub</a>
            </li>
          </ul>
          <a className="btn btn--gold btn--sm" href="/new/">
            Upload a spreadsheet
          </a>
        </nav>

        <div className="hero container" id="main">
          <div className="hero__text">
            <h1>Machine learning without learning machine learning</h1>
            <p className="lead hero__lead">Drop in a spreadsheet and get the best model, in plain English.</p>
            <div className="hero__actions">
              <a className="btn btn--gold" href="/new/">
                Upload a spreadsheet
                <Arrow />
              </a>
              <CopyCommand command="pip install plainml" tone="band" />
            </div>
            <p className="hero__runs">Free and open source. The website needs nothing installed.</p>
          </div>
          <figure className="hero__visual">
            <SceneStage
              className="stage--hero"
              visual={heroVisual}
              tone="band"
              spin
              far={1.3}
              still="/img/still-hero.jpg"
              label="plainml's 3D comparison of eight models predicting which customers leave, beside a ninth, shorter bar for a simple guess"
            />
            <figcaption className="readout readout--band">
              <span className="readout__file">churn.csv</span>
              <span>
                {best ? `${best.label} comes out on top. ` : ""}On customers it never saw it scores {TEST.toFixed(2)} out of 1
                {baseline != null ? `; a simple guess scores ${baseline.toFixed(2)}.` : "."} Drag to turn it.
              </span>
            </figcaption>
          </figure>
        </div>
      </header>

      <main>
        <section className="row container" aria-labelledby="ba-h">
          <div className="row__text">
            <h2 id="ba-h">A spreadsheet in, answers out</h2>
            <p className="lead">
              Here is churn.csv: 1,500 customers, and whether each one left. plainml compared 8 models and wrote down what
              it found.
            </p>
          </div>
          <div className="row__proof">
            <BeforeAfter />
          </div>
        </section>

        <section className="row row--lead container" id="tested" aria-labelledby="tested-h">
          <div className="row__text">
            <h2 id="tested-h">Tested on rows it never saw</h2>
            <p className="lead">
              Before training, plainml locks a fifth of your rows away. Only the winning model sees them, once, so the score
              you get is honest.
            </p>
            <a className="more" href="https://plainml-tpua.vercel.app/docs/how-it-works/">
              Read how plainml tests a model
            </a>
          </div>
          <div className="row__proof">
            <HeldOut />
          </div>
        </section>

        <section className="row container" id="jobs" aria-labelledby="jobs-h">
          <div className="row__text">
            <h2 id="jobs-h">Eight jobs from one upload</h2>
            <p className="lead">Pick one. Five show their real result in 3D; drag to turn it.</p>
          </div>
          <div className="row__proof">
            <Jobs />
          </div>
        </section>

        <section className="row container" id="where" aria-labelledby="where-h">
          <div className="row__text">
            <h2 id="where-h">Runs in your browser, or on your machine</h2>
            <p className="lead">
              On the website, Python runs inside the tab, so your file is never uploaded. It reads CSV, Excel, Parquet, JSON
              and TSV.
            </p>
            <a className="more" href="https://plainml-tpua.vercel.app/docs/website/#in-the-browser-with-no-server">
              Read how browser mode works
            </a>
          </div>
          <ol className="row__proof ways">
            <li className="way">
              <h3>In your browser</h3>
              <p>Python, pandas, NumPy, scikit-learn, LightGBM and XGBoost load into the page.</p>
              <ul className="way__icons" aria-label="What loads into the page">
                <li>
                  <img src="/icons/python.svg" alt="Python" width="28" height="28" />
                </li>
                <li>
                  <img src="/icons/scikitlearn.svg" alt="scikit-learn" width="28" height="28" />
                </li>
                <li>
                  <img src="/icons/pandas.svg" alt="pandas" width="28" height="28" />
                </li>
                <li>
                  <img src="/icons/numpy.svg" alt="NumPy" width="28" height="28" />
                </li>
              </ul>
            </li>
            <li className="way">
              <h3>On your computer</h3>
              <p>The same website, served from your machine.</p>
              <pre className="code">
                <code>
                  <span className="code__prompt">$ </span>pip install plainml{"\n"}
                  <span className="code__prompt">$ </span>plainml web
                </code>
              </pre>
            </li>
            <li className="way">
              <h3>In your own work</h3>
              <p>Train from the command line, then serve the model as an API or a Docker image.</p>
              <pre className="code">
                <code>
                  <span className="code__prompt">$ </span>plainml train churn.csv --target churned{"\n"}
                  <span className="code__prompt">$ </span>plainml serve
                </code>
              </pre>
            </li>
          </ol>
        </section>

        <section className="row container" id="questions" aria-labelledby="q-h">
          <div className="row__text">
            <h2 id="q-h">Questions</h2>
            <a className="more" href="https://github.com/pranay-obla/plainml/issues">
              Ask one on GitHub
            </a>
          </div>
          <dl className="row__proof faq">
            <div>
              <dt>Does it cost anything?</dt>
              <dd>No. plainml is free and open source under the MIT licence.</dd>
            </div>
            <div>
              <dt>Where does my spreadsheet go?</dt>
              <dd>Nowhere. On the website it stays in your browser tab; with plainml web it stays on your machine.</dd>
            </div>
            <div>
              <dt>How big can it be?</dt>
              <dd>Up to 200 MB in the browser. The command line also reads URLs and databases.</dd>
            </div>
            <div>
              <dt>When is something else better?</dt>
              <dd>For the last fraction of accuracy on a very large dataset, AutoGluon or a cloud platform will do better.</dd>
            </div>
          </dl>
        </section>
      </main>

      <footer className="letter">
        <div className="letter__grid container">
          <div className="letter__note">
            <p className="letter__text">
              Every AutoML tool I found did only part of the job, needed setting up first, or assumed you already knew machine
              learning. I wanted one that does all of it, with nothing to install and nothing to learn first. So I built
              plainml, for people who want the answers without learning machine learning first.
            </p>
            <p className="letter__sign">
              <span className="letter__name">Pranay Obla</span>
              <a href={ME}>github.com/pranay-obla</a>
              <span>November 2026</span>
            </p>
          </div>
          <div className="letter__try">
            <img className="letter__shot" src="/img/app-run.jpg" alt="plainml's results page for churn.csv: the test score, the best model, the baseline and the key findings" width="1280" height="560" loading="lazy" />
            <div className="letter__actions">
              <a className="btn btn--gold" href="/new/">
                Upload a spreadsheet
                <Arrow />
              </a>
              <CopyCommand command="pip install plainml" />
            </div>
          </div>
        </div>
        <div className="colophon container">
          <ul>
            <li>
              <a href="https://github.com/pranay-obla/plainml">Source on GitHub</a>
            </li>
            <li>
              <a href="https://pypi.org/project/plainml/">PyPI</a>
            </li>
            <li>
              <a href="https://plainml-tpua.vercel.app/docs/">Documentation</a>
            </li>
            <li>
              <a href="https://plainml-tpua.vercel.app/docs/commands/">Commands</a>
            </li>
            <li>
              <a href="https://plainml-tpua.vercel.app/docs/python-api/">Python API</a>
            </li>
            <li>
              <a href="https://github.com/pranay-obla/plainml/releases">Releases</a>
            </li>
            <li>
              <a href="/runs/">Your runs</a>
            </li>
            <li>
              <a href="/predict/">Predict with a saved model</a>
            </li>
          </ul>
          <p>MIT licence. Product names and logos belong to their owners.</p>
        </div>
      </footer>
    </div>
  );
}
