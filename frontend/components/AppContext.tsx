"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api, ApiError, getBridge, isBrowserMode, setUnauthorizedHandler, type BootState } from "@/lib/api";
import type { Info } from "@/lib/types";

interface Toast {
  id: number;
  message: string;
  hint?: string | null;
  bad?: boolean;
}

interface AppState {
  info: Info | null;
  infoError: ApiError | null;
  browser: boolean;
  boot: BootState | null;
  signedIn: boolean;
  setSignedIn: (value: boolean) => void;
  toast: (message: string, hint?: string | null, bad?: boolean) => void;
  fail: (error: unknown) => void;
}

const AppContext = createContext<AppState | null>(null);

export function useApp(): AppState {
  const state = useContext(AppContext);
  if (!state) throw new Error("useApp needs <AppProvider>");
  return state;
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [info, setInfo] = useState<Info | null>(null);
  const [infoError, setInfoError] = useState<ApiError | null>(null);
  const [browser, setBrowser] = useState(false);
  const [boot, setBoot] = useState<BootState | null>(null);
  const [signedIn, setSignedIn] = useState(true);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const next = useRef(1);

  const toast = useCallback((message: string, hint?: string | null, bad = false) => {
    const id = next.current++;
    setToasts((list) => [...list, { id, message, hint, bad }]);
    setTimeout(() => setToasts((list) => list.filter((t) => t.id !== id)), bad ? 9000 : 4500);
  }, []);

  const fail = useCallback(
    (error: unknown) => {
      const e = error instanceof ApiError ? error : new ApiError(String((error as Error)?.message || error));
      toast(e.message, e.hint, true);
    },
    [toast],
  );

  useEffect(() => {
    const inBrowser = isBrowserMode();
    setBrowser(inBrowser);
    const bridge = getBridge();
    const stop = bridge?.onBoot(setBoot);
    setUnauthorizedHandler(() => setSignedIn(false));
    api<Info>("/api/info")
      .then((answer) => {
        setInfo(answer);
        setSignedIn(!answer.needs_token || answer.signed_in);
      })
      .catch((error: ApiError) =>
        setInfoError(
          inBrowser
            ? new ApiError("Couldn't start plainml in this browser.", error.hint || error.message)
            : new ApiError("Can't reach the plainml app.", "Start it with: plainml web"),
        ),
      );
    return () => {
      stop?.();
    };
  }, []);

  const value = useMemo(
    () => ({ info, infoError, browser, boot, signedIn, setSignedIn, toast, fail }),
    [info, infoError, browser, boot, signedIn, toast, fail],
  );

  return (
    <AppContext.Provider value={value}>
      {children}
      <div id="toasts" aria-live="polite">
        {toasts.map((t) => (
          <div key={t.id} className={t.bad ? "toast bad" : "toast"} role={t.bad ? "alert" : "status"}>
            {t.message}
            {t.hint ? <small>{t.hint}</small> : null}
          </div>
        ))}
      </div>
    </AppContext.Provider>
  );
}
