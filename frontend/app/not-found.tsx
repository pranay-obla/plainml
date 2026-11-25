import { AppProvider } from "@/components/AppContext";
import "@/app/globals.css";
import { Footer, Header } from "@/components/Header";

export default function NotFound() {
  return (
    <AppProvider>
      <Header />
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
      <Footer />
    </AppProvider>
  );
}
