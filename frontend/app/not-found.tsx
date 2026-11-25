// The home page is the static public/index.html, so this is a plain link (a full page load).
export default function NotFound() {
  return (
    <main className="page">
      <div className="empty">
        <h1>Page not found</h1>
        <p className="sub">There&apos;s nothing at this address.</p>
        <p style={{ marginTop: 20 }}>
          <a className="btn primary" href="/">
            Go to plainml
          </a>
        </p>
      </div>
    </main>
  );
}
