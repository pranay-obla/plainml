import { AppProvider } from "@/components/AppContext";
import "@/app/globals.css";
import { Gate } from "@/components/Gate";
import { Footer, Header } from "@/components/Header";

// The app's pages (/new/, /analyze/, /runs/, /run/, /jobs/, /predict/): its header, the in-browser
// Python gate and its footer. The home page ((site)) has its own chrome.
export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <AppProvider>
      <Header />
      <Gate>{children}</Gate>
      <Footer />
    </AppProvider>
  );
}
