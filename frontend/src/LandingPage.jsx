import "./landing.css";
import logo from "./assets/logo.png";

export default function LandingPage({ onEnter }) {
  return (
    <div className="gw-landing">
      <div className="gw-bg">
        <div className="gw-aurora" />
        <div className="gw-grid" />

        {/* Decorative shapes */}
        <div className="gw-shapes" aria-hidden="true">
          <span className="gw-shape gw-shape-ring" />
          <span className="gw-shape gw-shape-blob" />
          <span className="gw-shape gw-shape-chip" />
          <span className="gw-shape gw-shape-lines" />
        </div>

        <div className="gw-noise" />
        <div className="gw-vignette" />
      </div>

      <header className="gw-nav">
        <div className="gw-brand">
          <img className="gw-brand-logo" src={logo} alt="GrowWiseAI logo" />
          <span className="gw-brand-name">GrowWiseAI</span>
        </div>
      </header>

      <main className="gw-hero">
        <div className="gw-hero-inner">
          <div className="gw-logo-wrap" aria-hidden="true">
            <img className="gw-hero-logo" src={logo} alt="" />
          </div>

          <h1 className="gw-title">
            GrowWiseAI<span className="gw-title-accent">.</span>
          </h1>

          <p className="gw-slogan">
            “not every soil can bear all things. Be practical”
          </p>

          <p className="gw-subtitle">
            Pick a spot in the lower 48 and we’ll pull the local soil + climate
            conditions for you. Adjust anything you want, then run a prediction
            to see how suitable that location is for tree survival.
          </p>

          <div className="gw-cta">
            <button
              className="gw-btn gw-btn-primary"
              onClick={onEnter}
              type="button"
            >
              Launch Demo
            </button>

            <a
              className="gw-btn gw-btn-ghost"
              href="#how"
              onClick={(e) => {
                e.preventDefault();
                const el = document.getElementById("how");
                if (el) el.scrollIntoView({ behavior: "smooth" });
              }}
            >
              How it works
            </a>
          </div>

          <div className="gw-mini">
            <div className="gw-mini-item">
              <div className="gw-mini-num">01</div>
              <div className="gw-mini-text">Pick a point (lower-48 only)</div>
            </div>
            <div className="gw-mini-item">
              <div className="gw-mini-num">02</div>
              <div className="gw-mini-text">Adjust soil + climate inputs</div>
            </div>
            <div className="gw-mini-item">
              <div className="gw-mini-num">03</div>
              <div className="gw-mini-text">Run the model + read survivability</div>
            </div>
          </div>
        </div>

        <div className="gw-preview">
          <div className="gw-preview-card">
            <div className="gw-preview-top">
              {/* macOS order: red, yellow, green */}
              <div className="gw-dot gw-dot-r" />
              <div className="gw-dot gw-dot-y" />
              <div className="gw-dot gw-dot-g" />
              <span className="gw-preview-title">Preview</span>
            </div>

            <div className="gw-preview-body">
              <div className="gw-preview-map">
                <div className="gw-preview-map-label">Contiguous USA</div>
              </div>

              <div className="gw-preview-panel">
                <div className="gw-skel gw-skel-lg" />
                <div className="gw-skel" />
                <div className="gw-skel" />
                <div className="gw-skel gw-skel-md" />
                <div className="gw-preview-btn">Run prediction</div>
              </div>
            </div>

            <div className="gw-preview-foot">
              Built with React + Leaflet • FastAPI backend • ML inference
            </div>
          </div>
        </div>
      </main>

      <section className="gw-how" id="how">
        <div className="gw-how-inner">
          <h2 className="gw-how-title">How GrowWiseAI works</h2>

          <p className="gw-how-lead">
            It’s a simple flow: pick a place, pull real environmental inputs,
            test a few “what ifs,” and run the model.
          </p>

          <div className="gw-how-grid">
            <div className="gw-how-card">
              <div className="gw-how-num">01</div>
              <div className="gw-how-head">Click a location (lower-48 only)</div>
              <div className="gw-how-text">
                The map only accepts clicks inside the contiguous U.S. Outside
                the boundary you’ll see a red ❌ and clicks are ignored.
              </div>
            </div>

            <div className="gw-how-card">
              <div className="gw-how-num">02</div>
              <div className="gw-how-head">We fetch baseline conditions</div>
              <div className="gw-how-text">
                Your click sends <span className="gw-code">lat/lon</span> to{" "}
                <span className="gw-code">/api/fetch-features</span>, returning
                elevation, temperature, humidity, and soil nutrients.
              </div>
            </div>

            <div className="gw-how-card">
              <div className="gw-how-num">03</div>
              <div className="gw-how-head">You can override any values</div>
              <div className="gw-how-text">
                Adjust sliders to explore scenarios. A baseline copy is kept so
                everything can be restored instantly.
              </div>
            </div>

            <div className="gw-how-card">
              <div className="gw-how-num">04</div>
              <div className="gw-how-head">Run prediction + view results</div>
              <div className="gw-how-text">
                Press run to POST the current features to{" "}
                <span className="gw-code">/api/predict</span>. The model returns
                a survivability estimate shown in the results panel.
              </div>
            </div>
          </div>

          <div className="gw-how-cta">
            <button
              className="gw-btn gw-btn-primary"
              onClick={onEnter}
              type="button"
            >
              Launch Demo
            </button>
          </div>
        </div>
      </section>

      <footer className="gw-footer">
        <span>© {new Date().getFullYear()} GrowWiseAI</span>
        <span className="gw-footer-muted">CXC Hackathon</span>
      </footer>
    </div>
  );
}
