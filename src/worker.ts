import {
  AutoModelForCausalLM,
  AutoTokenizer,
  PreTrainedModel,
  PreTrainedTokenizer,
  Tensor,
  softmax,
  type ProgressCallback,
} from "@huggingface/transformers";

const MAX_TOKENS = 512;

// TODO: find hf built-in
const GEMMA_EOT_TOKEN_ID = 106;

let cancelCurrent: (() => void) | null = null;

function progressCallback(progressInfo: any): void {
  if (progressInfo.status === "progress") {
    console.log(`progress: ${progressInfo.progress}`);
  }
}

class TextGenerationPipeline {
  static model_id = "onnx-community/gemma-3-270m-it-ONNX";
  static tokenizer: Promise<PreTrainedTokenizer> | null = null;
  static model: Promise<PreTrainedModel> | null = null;

  static async getInstance(
    progress_callback: ProgressCallback | undefined = undefined
  ) {
    this.tokenizer ??= AutoTokenizer.from_pretrained(this.model_id, {
      progress_callback,
    });

    this.model ??= AutoModelForCausalLM.from_pretrained(this.model_id, {
      dtype: "fp32",
      device: "webgpu",
      progress_callback,
    });

    return Promise.all([this.tokenizer, this.model]);
  }
}

type WorkerMessage =
  | { type: "generate"; prompt: string }
  | { type: "stop" }
  | { type: "resume" }
  | { type: "step" };

type WorkerResponse =
  | { type: "ready" }
  | { type: "generation-start" }
  | { type: "token"; text: string }
  | { type: "done" }
  | { type: "error"; error: string };

function idsToTensor(ids: number[]): Tensor {
  const data = BigInt64Array.from(ids, (value) => BigInt(value));
  return new Tensor("int64", data, [1, ids.length]);
}

function sampleFromProbs(probs: number[]): number {
  const r = Math.random();
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (r <= acc) return i;
  }
  return probs.length - 1;
}

class Worker {
  private tokenizer!: PreTrainedTokenizer;
  private model!: PreTrainedModel;
  private tokens: number[] = [];

  // indicates whether the worker is in the middle of generation.
  // still true if the generation is paused.
  private inProgress = false;

  constructor() {
    TextGenerationPipeline.getInstance().then(([tokenizer, model]) => {
      this.tokenizer = tokenizer;
      this.model = model;
    });
  }

  /**
   * Generates a single token, updating this.tokens and sending the token to the main thread.
   */
  private async generateSingle(): Promise<number> {
    const input_ids = idsToTensor(this.tokens);
    const attention_mask = idsToTensor(Array(this.tokens.length).fill(1));
    const out = await this.model({
      input_ids,
      attention_mask,
    });

    const nextProbs = softmax(out.logits[0][out.logits.dims[1] - 1].data);
    const nextToken = sampleFromProbs(nextProbs);
    console.log("[worker]: nextToken", nextToken);

    this.tokens = [...this.tokens, nextToken];

    const decoded = this.tokenizer.decode([nextToken], {
      skip_special_tokens: true,
    });

    self.postMessage({
      type: "token",
      text: decoded,
    } satisfies WorkerResponse);

    return nextToken;
  }

  /**
   * Generates tokens until the maximum number of tokens is reached or the generation is cancelled.
   * Assumes that this.tokens is already set by a previous generation.
   */
  private async generateLoop() {
    if (this.tokens.length === 0) {
      throw new Error("generate() should be called first.");
    }

    cancelCurrent?.();
    let cancelled = false;
    cancelCurrent = () => {
      cancelled = true;
    };

    while (!cancelled && this.tokens.length < MAX_TOKENS) {
      const nextToken = await this.generateSingle();
      if (
        nextToken === GEMMA_EOT_TOKEN_ID ||
        this.tokens.length >= MAX_TOKENS
      ) {
        self.postMessage({ type: "done" } satisfies WorkerResponse);
        this.inProgress = false;
        return;
      }
    }
  }

  public async generate(prompt: string) {
    this.inProgress = true;

    const inputIds = this.tokenizer.apply_chat_template(
      [{ role: "user", content: prompt }],
      { tokenize: true, return_tensor: false, add_generation_prompt: true }
    ) as number[];

    this.tokens = [...inputIds];
    console.log("[worker]: tokens", this.tokens);

    self.postMessage({
      type: "generation-start",
    } satisfies WorkerResponse);

    await this.generateLoop();
  }

  public async resume() {
    if (!this.inProgress) {
      return;
    }

    await this.generateLoop();
  }

  public async step() {
    if (!this.inProgress) {
      return;
    }

    const nextToken = await this.generateSingle();
    if (nextToken === GEMMA_EOT_TOKEN_ID || this.tokens.length >= MAX_TOKENS) {
      self.postMessage({ type: "done" } satisfies WorkerResponse);
      this.inProgress = false;
      return;
    }
  }
}

let worker = new Worker();

self.addEventListener("message", (event: MessageEvent<WorkerMessage>) => {
  const data = event.data;
  if (data.type === "generate") {
    worker.generate(data.prompt).catch((err) => {
      self.postMessage({
        type: "error",
        error: err instanceof Error ? err.message : String(err),
      } satisfies WorkerResponse);
    });
  } else if (data.type === "stop") {
    cancelCurrent?.();
  } else if (data.type === "resume") {
    worker.resume();
  } else if (data.type === "step") {
    worker.step();
  }
});

TextGenerationPipeline.getInstance()
  .then(() => {
    self.postMessage({ type: "ready" } satisfies WorkerResponse);
  })
  .catch((err) => {
    self.postMessage({
      type: "error",
      error: err instanceof Error ? err.message : String(err),
    } satisfies WorkerResponse);
  });

export {};
