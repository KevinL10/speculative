import { useEffect, useState } from "react";
import reactLogo from "./assets/react.svg";
import viteLogo from "/vite.svg";
import "./App.css";
import { pipeline } from "@huggingface/transformers";

// Create a feature-extraction pipeline
const extractor = await pipeline(
  "feature-extraction",
  "mixedbread-ai/mxbai-embed-xsmall-v1",
  { device: "webgpu" }
);

// Compute embeddings

function App() {
  const [count, setCount] = useState(0);
  useEffect(() => {
    const fetchEmbeddings = async () => {
      const texts = ["Hello world!", "This is an example sentence."];
      const embeddings = await extractor(texts, {
        pooling: "mean",
        normalize: true,
      });
      console.log(embeddings.tolist());
    };
    fetchEmbeddings();
  }, []);

  return (
    <>
      <div>
        <a href="https://vite.dev" target="_blank">
          <img src={viteLogo} className="logo" alt="Vite logo" />
        </a>
        <a href="https://react.dev" target="_blank">
          <img src={reactLogo} className="logo react" alt="React logo" />
        </a>
      </div>
      <h1>Vite + React</h1>
      <div className="card">
        <button onClick={() => setCount((count) => count + 1)}>
          count is {count}
        </button>
        <p>
          Edit <code>src/App.tsx</code> and save to test HMR
        </p>
      </div>
      <p className="read-the-docs">
        Click on the Vite and React logos to learn more
      </p>
    </>
  );
}

export default App;
