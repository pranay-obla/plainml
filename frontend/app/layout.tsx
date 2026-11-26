import type { Metadata, Viewport } from "next";
import { Bricolage_Grotesque, Instrument_Sans, JetBrains_Mono } from "next/font/google";

// plainml's faces: Bricolage Grotesque for headings, Instrument Sans for text, JetBrains Mono for code.
// The home page ((site)) and the app ((app)) share them; each group has its own layout, chrome and
// stylesheet (app/globals.css is the app's, imported in (app)/layout.tsx and not-found.tsx).
const display = Bricolage_Grotesque({ subsets: ["latin"], variable: "--font-bricolage", display: "swap" });
const body = Instrument_Sans({ subsets: ["latin"], variable: "--font-instrument", display: "swap" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-jetbrains", display: "swap" });

export const metadata: Metadata = {
  title: "plainml: machine learning without learning machine learning",
  description:
    "Free, open-source AutoML that trains, compares and explains models in plain English. Runs entirely in your browser: your data never leaves your computer.",
  icons: { icon: "/favicon.svg" },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#fff9ec" },
    { media: "(prefers-color-scheme: dark)", color: "#0c0505" },
  ],
};

// Applies the saved theme before the first paint (the HTML reports share the same setting).
const THEME_SCRIPT = `document.documentElement.dataset.js="1";try{var t=localStorage.getItem("plainml-theme");if(t==="light"||t==="dark")document.documentElement.dataset.theme=t}catch(e){}`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable} ${mono.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
