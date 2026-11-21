import type { Metadata, Viewport } from "next";
import { Bricolage_Grotesque, Instrument_Sans, JetBrains_Mono } from "next/font/google";
import { AppProvider } from "@/components/AppContext";
import { Gate } from "@/components/Gate";
import { Footer, Header } from "@/components/Header";
import "./globals.css";

const display = Bricolage_Grotesque({ subsets: ["latin"], variable: "--font-display", display: "swap" });
const body = Instrument_Sans({ subsets: ["latin"], variable: "--font-body", display: "swap" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono", display: "swap" });

export const metadata: Metadata = {
  title: "plainml: upload a spreadsheet, get a model you understand",
  description:
    "Free, open-source AutoML that trains, compares and explains models in plain English. Runs entirely in your browser: your data never leaves your computer.",
  icons: {
    icon: "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='9' fill='%23990000'/%3E%3Cpath d='M9 22V14M16 22V9M23 22v-5' stroke='%23ffcc00' stroke-width='3' stroke-linecap='round'/%3E%3C/svg%3E",
  },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#fff9ec" },
    { media: "(prefers-color-scheme: dark)", color: "#0c0505" },
  ],
};

// Applies the saved theme before the first paint (the HTML reports share the same setting).
const THEME_SCRIPT = `try{var t=localStorage.getItem("plainml-theme");if(t==="light"||t==="dark")document.documentElement.dataset.theme=t}catch(e){}`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable} ${mono.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body>
        <AppProvider>
          <Header />
          <Gate>{children}</Gate>
          <Footer />
        </AppProvider>
      </body>
    </html>
  );
}
