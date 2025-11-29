import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import "./App.css";

function App() {
  const [prompt, setPrompt] = useState("tell me about tokyo");
  const [generation, setGeneration] = useState<string | null>(null);
  const [verifyTokens, setVerifyTokens] = useState<string[]>([]);
  const [draftTokens, setDraftTokens] = useState<string[]>([]);
  const [rejectedTokens, setRejectedTokens] = useState<string[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isPaused, setIsPaused] = useState(true);
  const [isLoading, setIsLoading] = useState(true);
  const [loadingProgress, setLoadingProgress] = useState<{
    file: string;
    progress: number;
  } | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const verifyTokensRef = useRef<string[]>([]);
  const draftTokensRef = useRef<string[]>([]);

  useEffect(() => {
    verifyTokensRef.current = verifyTokens;
  }, [verifyTokens]);

  useEffect(() => {
    draftTokensRef.current = draftTokens;
  }, [draftTokens]);

  useEffect(() => {
    const worker = new Worker(new URL("./worker.ts", import.meta.url), {
      type: "module",
    });

    const handleMessage = (event: MessageEvent) => {
      const message = event.data;
      if (message.type === "generation-start") {
        setIsGenerating(true);
        setIsPaused(false);
        setGeneration("");
        setDraftTokens([]);
        setVerifyTokens([]);
        setRejectedTokens([]);
      } else if (message.type === "update") {
        console.log("[app]: update", message);

        if (message.stage === "draft") {
          setDraftTokens((prev) => [...prev, message.token]);
        } else if (message.stage === "verify") {
          // We verify from left to right, so remove the first draft token and move it to verified.
          console.log(message.token, draftTokens);
          setVerifyTokens((prev) => [...prev, message.token]);
          setDraftTokens((prev) => prev.slice(1));
        } else if (message.stage === "sample") {
          setGeneration(
            (prev) => prev + verifyTokensRef.current.join("") + message.token
          );
          setVerifyTokens([]);
          setDraftTokens([]);
          setRejectedTokens([]);
        } else if (message.stage === "reject") {
          setRejectedTokens((prev) => [...prev, ...draftTokensRef.current]);
          setDraftTokens([]);
        }
      } else if (message.type === "done") {
        setIsGenerating(false);
        setIsPaused(true);
      } else if (message.type === "error") {
        setGeneration(`Error: ${message.error ?? "Unknown error"}`);
      } else if (message.type === "ready") {
        console.log("worker ready");
        setIsLoading(false);
        setLoadingProgress(null);
      } else if (message.type === "loading-progress") {
        setIsLoading(true);
        setLoadingProgress({
          file: message.file,
          progress: message.progress,
        });
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
    setDraftTokens([]);
    setVerifyTokens([]);
    setRejectedTokens([]);
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

      <div className={`generation-section ${isLoading ? "loading" : ""}`}>
        {isLoading && loadingProgress ? (
          <div className="loading-container">
            <div className="loading-spinner"></div>
            <div className="loading-text">
              Loading {loadingProgress.file}...
            </div>
            <div className="loading-progress">
              {loadingProgress.progress.toFixed(2)}%
            </div>
          </div>
        ) : (
          (generation ||
            verifyTokens.length > 0 ||
            draftTokens.length > 0 ||
            rejectedTokens.length > 0) && (
            <div className="generation-text">
              {generation}
              {verifyTokens.length > 0 && (
                <span className="verified-tokens">{verifyTokens.join("")}</span>
              )}
              {rejectedTokens.length > 0 && (
                <span className="rejected-tokens">
                  {rejectedTokens.join("")}
                </span>
              )}
              {draftTokens.length > 0 && (
                <span className="draft-tokens">{draftTokens.join("")}</span>
              )}
            </div>
          )
        )}
      </div>

      <div className="nav-controls">
        <div className="control-bar">
          <button
            className="control-btn"
            onClick={() => {
              if (isPaused) {
                workerRef.current?.postMessage({
                  type: "resume",
                  prompt: prompt,
                });
              } else {
                workerRef.current?.postMessage({
                  type: "stop",
                  prompt: prompt,
                });
              }

              if (isGenerating) {
                setIsPaused(!isPaused);
              }
            }}
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
          <button
            className="control-btn"
            title="Next Step"
            onClick={() => {
              workerRef.current?.postMessage({ type: "step" });
            }}
          >
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
