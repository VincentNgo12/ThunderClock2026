// Connects to the backend telemetry WebSocket, reconnects on loss and reports the
// latest snapshot plus a measured message rate.

import { useEffect, useRef, useState } from "react";
import type { WorkerTelemetry } from "../telemetry/types";

export type SocketState = "connecting" | "open" | "closed";

const RECONNECT_DELAY_MS = 1000;

export function telemetryUrl(): string {
  // Override with ?ws=ws://host:port/ws/telemetry or VITE_TELEMETRY_URL.
  const fromQuery = new URLSearchParams(window.location.search).get("ws");
  return fromQuery ?? import.meta.env.VITE_TELEMETRY_URL ?? "ws://127.0.0.1:8000/ws/telemetry";
}

export function useTelemetry(url: string) {
  const [socketState, setSocketState] = useState<SocketState>("connecting");
  const [latest, setLatest] = useState<WorkerTelemetry | null>(null);
  const [rateHz, setRateHz] = useState(0);
  const [lastMessageAt, setLastMessageAt] = useState<number | null>(null);
  const messageCount = useRef(0);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let retryTimer: number | undefined;
    let stopped = false;

    const connect = () => {
      setSocketState("connecting");
      socket = new WebSocket(url);
      socket.onopen = () => setSocketState("open");
      socket.onmessage = (event) => {
        try {
          setLatest(JSON.parse(event.data) as WorkerTelemetry);
          setLastMessageAt(performance.now());
          messageCount.current += 1;
        } catch {
          // Ignore malformed frames; the next snapshot replaces this one anyway.
        }
      };
      socket.onclose = () => {
        setSocketState("closed");
        if (!stopped) {
          retryTimer = window.setTimeout(connect, RECONNECT_DELAY_MS);
        }
      };
    };
    connect();

    // Count messages over each second to show the live update rate.
    const rateTimer = window.setInterval(() => {
      setRateHz(messageCount.current);
      messageCount.current = 0;
    }, 1000);

    return () => {
      stopped = true;
      window.clearTimeout(retryTimer);
      window.clearInterval(rateTimer);
      socket?.close();
    };
  }, [url]);

  return { socketState, latest, rateHz, lastMessageAt };
}
