"use client";

import { useState, type FormEvent, type ReactNode } from "react";
import { api } from "@/lib/api";
import { useApp } from "./AppContext";
import { Icon } from "./Icon";

/** Asks for the access token when `plainml web --token` started the app, and shows startup errors. */
export function Gate({ children }: { children: ReactNode }) {
  const { info, infoError, signedIn } = useApp();
  if (infoError && !info) {
    return (
      <main className="page">
        <div className="empty">
          <div className="orb">
            <Icon name="alert" />
          </div>
          <h2>{infoError.message}</h2>
          {infoError.hint ? <p className="sub">{infoError.hint}</p> : null}
        </div>
      </main>
    );
  }
  if (info && info.needs_token && !signedIn) return <SignIn />;
  return <>{children}</>;
}

function SignIn() {
  const { setSignedIn, fail } = useApp();
  const [token, setToken] = useState("");
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    try {
      await api("/api/login", { json: { token } });
      setSignedIn(true);
    } catch (error) {
      fail(error);
    }
  };
  return (
    <main className="page">
      <form className="panel pad signin stack" onSubmit={submit}>
        <h2>Sign in</h2>
        <p className="sub">This plainml app was started with an access token. Ask whoever runs it.</p>
        <input
          className="input"
          type="password"
          autoComplete="current-password"
          placeholder="Access token"
          aria-label="Access token"
          value={token}
          onChange={(e) => setToken(e.target.value)}
          autoFocus
        />
        <button className="btn primary" type="submit">
          Continue
        </button>
      </form>
    </main>
  );
}
