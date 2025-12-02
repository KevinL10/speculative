import {
  useEffect,
  useRef,
  useState,
  useCallback,
  type KeyboardEvent,
} from "react";
import "./App.css";

function App() {
  const [prompt, setPrompt] = useState("tell me a story");
  const [generation, setGeneration] = useState<string | null>(null);
  const [verifyTokens, setVerifyTokens] = useState<string[]>([]);
  const [draftTokens, setDraftTokens] = useState<string[]>([]);
  const [rejectedTokens, setRejectedTokens] = useState<string[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isPaused, setIsPaused] = useState(true);
  const [isLoading, setIsLoading] = useState(true);
  const [loadingProgress, setLoadingProgress] = useState<
    Record<string, number>
  >({});
  const workerRef = useRef<Worker | null>(null);
  const verifyTokensRef = useRef<string[]>([]);
  const draftTokensRef = useRef<string[]>([]);
  const currentPromptIdRef = useRef<number>(0);

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
      } else if (
        message.type === "update" &&
        message.promptId === currentPromptIdRef.current
      ) {
        if (message.stage === "draft") {
          setDraftTokens((prev) => [...prev, message.token]);
        } else if (message.stage === "verify") {
          // We verify from left to right, so remove the first draft token and move it to verified.
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
      } else if (
        message.type === "done" &&
        message.promptId === currentPromptIdRef.current
      ) {
        console.log("[app]: done");
        setIsGenerating(false);
        setIsPaused(true);
        setGeneration((prev) => prev + verifyTokensRef.current.join(""));
        setVerifyTokens([]);
        setDraftTokens([]);
        setRejectedTokens([]);
      } else if (message.type === "error") {
        setGeneration(`Error: ${message.error ?? "Unknown error"}`);
      } else if (message.type === "ready") {
        setIsLoading(false);
        setLoadingProgress({});
      } else if (message.type === "loading-progress") {
        setLoadingProgress((prev) => {
          const updated = { ...prev };
          if (message.progress >= 100.0) {
            delete updated[message.file];
          } else {
            updated[message.file] = message.progress;
          }
          return updated;
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

  const startGeneration = useCallback(() => {
    const content = prompt.trim();
    if (!content) return;

    currentPromptIdRef.current += 1;

    setGeneration("");
    setDraftTokens([]);
    setVerifyTokens([]);
    setRejectedTokens([]);
    workerRef.current?.postMessage({ type: "stop" });
    workerRef.current?.postMessage({
      type: "generate",
      prompt: content,
      promptId: currentPromptIdRef.current,
    });
  }, [prompt]);

  const togglePauseResume = useCallback(() => {
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
  }, [isPaused, isGenerating, prompt]);

  const handleStartOrToggle = useCallback(() => {
    if (!isGenerating) {
      startGeneration();
    } else {
      togglePauseResume();
    }
  }, [isGenerating, startGeneration, togglePauseResume]);

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") startGeneration();
  };

  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      if (
        e.key === " " &&
        e.target instanceof HTMLElement &&
        e.target.tagName !== "INPUT"
      ) {
        e.preventDefault();
        handleStartOrToggle();
      }
    };

    window.addEventListener("keydown", handleGlobalKeyDown as any);
    return () => {
      window.removeEventListener("keydown", handleGlobalKeyDown as any);
    };
  }, [handleStartOrToggle]);

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
        {isLoading ? (
          <div className="loading-container">
            <div className="loading-spinner"></div>
            {Object.keys(loadingProgress).length > 0 ? (
              Object.entries(loadingProgress).map(([file, progress]) => (
                <div key={file} className="loading-item">
                  <div className="loading-text">Fetching {file}</div>
                  <div className="loading-progress">{progress.toFixed(2)}%</div>
                </div>
              ))
            ) : (
              <div className="loading-item">
                <div className="loading-text">
                  Loading models (may take a few seconds)...
                </div>
              </div>
            )}
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
          {!isGenerating && (
            <button
              className="control-btn"
              onClick={startGeneration}
              title="Start"
            >
              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="currentColor"
              >
                <path d="M8 5v14l11-7z" />
              </svg>
            </button>
          )}
          {isGenerating && (
            <button
              className="control-btn"
              onClick={togglePauseResume}
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
          )}
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
