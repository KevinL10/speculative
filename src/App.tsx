import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import "./App.css";

function App() {
  const [prompt, setPrompt] = useState("");
  const [generation, setGeneration] = useState<string | null>(null);
  const [isPaused, setIsPaused] = useState(false);
  const workerRef = useRef<Worker | null>(null);

  useEffect(() => {
    const worker = new Worker(new URL("./worker.ts", import.meta.url), {
      type: "module",
    });

    const handleMessage = (event: MessageEvent) => {
      const message = event.data;
      if (message.type === "generation-start") {
        setGeneration("");
      } else if (message.type === "token") {
        setGeneration((prev) => (prev ? prev + message.text : message.text));
      } else if (message.type === "done") {
        console.log("finisehd generating");
      } else if (message.type === "error") {
        setGeneration(`Error: ${message.error ?? "Unknown error"}`);
      } else if (message.type === "ready") {
        console.log("worker ready");
      }
    };

    workerRef.current = worker;
    worker.addEventListener("message", handleMessage);

    return () => {
      worker.removeEventListener("message", handleMessage);
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  const startGeneration = () => {
    const content = prompt.trim();
    if (!content) return;
    setGeneration("");
    workerRef.current?.postMessage({ type: "stop" });
    workerRef.current?.postMessage({ type: "generate", prompt: content });
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") startGeneration();
  };

  return (
    <div className="app-container">
      <div className="input-section">
        <input
          type="text"
          className="prompt-input"
          placeholder="Enter your prompt..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={handleKeyDown}
        />
      </div>

      <div className="generation-section">
        {generation && <div className="generation-text">{generation}</div>}
      </div>

      <div className="nav-controls">
        <div className="control-bar">
          <button
            className="control-btn"
            onClick={() => setIsPaused(!isPaused)}
            title={isPaused ? "Resume" : "Pause"}
          >
            {isPaused ? (
              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="currentColor"
              >
                <path d="M8 5v14l11-7z" />
              </svg>
            ) : (
              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="currentColor"
              >
                <rect x="6" y="4" width="4" height="16" />
                <rect x="14" y="4" width="4" height="16" />
              </svg>
            )}
          </button>
          <button className="control-btn" title="Next Step">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
              <path d="M6 18l8.5-6L6 6v12z" />
              <rect x="16" y="6" width="2" height="12" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
