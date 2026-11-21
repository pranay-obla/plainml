import Link from "next/link";

export default function NotFound() {
  return (
    <main className="page">
      <div className="empty">
        <h1>Page not found</h1>
        <p className="sub">There&apos;s nothing at this address.</p>
        <p style={{ marginTop: 20 }}>
          <Link className="btn primary" href="/">
            Go to plainml
          </Link>
        </p>
      </div>
    </main>
  );
}
