import {
  AutoModelForCausalLM,
  AutoTokenizer,
  PreTrainedModel,
  PreTrainedTokenizer,
  Tensor,
  softmax,
} from "@huggingface/transformers";

const LOOKAHEAD = 5;
// Note: this is the Gemma <end_of_turn> token.
// TODO: fix token id handling
const EOT_TOKEN_ID = 106;

let cancelCurrent: (() => void) | null = null;

function createProgressCallback(modelName: string) {
  return (progressInfo: any): void => {
    if (progressInfo.status === "progress") {
      self.postMessage({
        type: "loading-progress",
        file: `${modelName}: ${progressInfo.file || "model"}`,
        progress: progressInfo.progress || 0,
      });
    }
  };
}

class TextGenerationPipeline {
  // works with q4 but not q4f16.
  static draftModelId = "onnx-community/gemma-3-270m-it-ONNX";
  static targetModelId = "onnx-community/gemma-3-1b-it-ONNX-GQA";

  static tokenizer: Promise<PreTrainedTokenizer> | null = null;
  static draftModel: Promise<PreTrainedModel> | null = null;
  static targetModel: Promise<PreTrainedModel> | null = null;

  static async getInstance() {
    this.tokenizer ??= AutoTokenizer.from_pretrained(this.draftModelId, {
      progress_callback: createProgressCallback("Tokenizer"),
    });

    this.draftModel ??= AutoModelForCausalLM.from_pretrained(
      this.draftModelId,
      {
        dtype: "q4",
        device: "webgpu",
        progress_callback: createProgressCallback(
          "Draft Model (gemma-3-270m-it)"
        ),
      }
    ).catch((err) => {
      console.error("Error loading draft model:", err);
      throw err;
    });

    this.targetModel ??= AutoModelForCausalLM.from_pretrained(
      this.targetModelId,
      {
        dtype: "q4f16",
        device: "webgpu",
        progress_callback: createProgressCallback(
          "Target Model (gemma-3-1b-it)"
        ),
      }
    ).catch((err) => {
      console.error("Error loading target model:", err);
      throw err;
    });

    return Promise.all([this.tokenizer, this.draftModel, this.targetModel]);
  }
}

type GenerationStage = "draft" | "verify" | "sample";

type WorkerMessage =
  | { type: "generate"; prompt: string; promptId: number }
  | { type: "stop" }
  | { type: "resume" }
  | { type: "step" };

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
  private draftModel!: PreTrainedModel;
  private targetModel!: PreTrainedModel;
  private tokens: number[] = [];
  private state: GenerationStage = "draft";

  private draftTokens: number[] = [];
  private draftProbs: number[][] = [];
  private targetProbs: number[][] = [];
  private verifyIdx: number | null = null;
  private currentPromptId: number | null = null;

  // indicates whether the worker is in the middle of generation.
  // still true if the generation is paused.
  private inProgress = false;

  constructor() {
    TextGenerationPipeline.getInstance().then(
      ([tokenizer, draftModel, targetModel]) => {
        this.tokenizer = tokenizer;
        this.draftModel = draftModel;
        this.targetModel = targetModel;
      }
    );
  }

  /**
   * Returns the logits from a forward pass.
   */
  private async sample(model: PreTrainedModel): Promise<any> {
    const input_ids = idsToTensor([...this.tokens, ...this.draftTokens]);
    const attention_mask = idsToTensor(
      Array(this.tokens.length + this.draftTokens.length).fill(1)
    );
    const out = await model({
      input_ids,
      attention_mask,
    });
    // await new Promise((resolve) => setTimeout(resolve, 100));
    return out.logits;
  }

  private async sampleTokenAndProbs(
    model: PreTrainedModel
  ): Promise<[number, number[]]> {
    const logits = await this.sample(model);
    const probs = softmax(logits[0][logits.dims[1] - 1].data);
    return [sampleFromProbs(probs), probs];
  }

  private async sampleTargetProbs(): Promise<number[][]> {
    const logits = await this.sample(this.targetModel);
    const probs = [];
    for (let t = 0; t < this.draftTokens.length + 1; t++) {
      probs.push(softmax(logits[0][this.tokens.length + t - 1].data));
    }
    return probs;
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

    while (!cancelled && this.tokens[this.tokens.length - 1] !== EOT_TOKEN_ID) {
      try {
        await this.step();
      } catch (err) {
        console.error("[worker]: error in step", err);
        self.postMessage({
          type: "error",
          error: err instanceof Error ? err.message : String(err),
          promptId: this.currentPromptId!,
        });
        break;
      }
    }
  }

  public async generate(prompt: string, promptId: number) {
    cancelCurrent?.();
    this.inProgress = true;
    this.currentPromptId = promptId;
    this.draftTokens = [];
    this.draftProbs = [];
    this.targetProbs = [];
    this.verifyIdx = null;
    this.state = "draft";

    const inputIds = this.tokenizer.apply_chat_template(
      [
        {
          role: "system",
          content:
            "You are a helpful assistant. Respond directly and concisely.",
        },
        { role: "user", content: prompt },
      ],
      { tokenize: true, return_tensor: false, add_generation_prompt: true }
    ) as number[];
    this.tokens = [...inputIds];

    self.postMessage({
      type: "generation-start",
      promptId: this.currentPromptId!,
    });

    await this.generateLoop();
  }

  public async resume() {
    if (!this.inProgress) {
      return;
    }

    await this.generateLoop();
  }

  private async handleDraft() {
    const [nextToken, nextProbs] = await this.sampleTokenAndProbs(
      this.draftModel
    );
    this.draftTokens.push(nextToken);
    this.draftProbs.push(nextProbs);

    if (this.draftTokens.length === LOOKAHEAD || nextToken === EOT_TOKEN_ID) {
      this.state = "verify";
    }

    const decoded = this.tokenizer.decode([nextToken], {
      skip_special_tokens: true,
    });

    self.postMessage({
      type: "update",
      stage: "draft",
      token: decoded,
      promptId: this.currentPromptId!,
    });
  }

  private async handleVerify() {
    // On the first time we verify a token, we need to sample the probs from the target model.
    if (this.verifyIdx === null) {
      this.verifyIdx = 0;
      this.targetProbs = await this.sampleTargetProbs();
    }

    // Verify the token
    const r = Math.random();
    const x = this.draftTokens[this.verifyIdx];
    const q = this.targetProbs[this.verifyIdx][x];
    const p = this.draftProbs[this.verifyIdx][x];
    const decoded = this.tokenizer.decode([x], {
      skip_special_tokens: true,
    });

    if (r < Math.min(1, q / p)) {
      console.log(`[worker]: accepted ${decoded} (${x})`);
      this.tokens.push(x);
      self.postMessage({
        type: "update",
        stage: "verify",
        token: decoded,
        promptId: this.currentPromptId!,
      });
    } else {
      console.log(`[worker]: rejected ${decoded} (${x})`);
      self.postMessage({
        type: "update",
        stage: "reject",
        token: decoded,
        promptId: this.currentPromptId!,
      });
      this.state = "sample";
      await new Promise((resolve) => setTimeout(resolve, 100));
      return;
    }

    this.verifyIdx++;
    if (this.verifyIdx === this.draftTokens.length) {
      this.state = "sample";
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  private async handleSample() {
    let probs: number[] = [];
    if (this.verifyIdx === this.draftTokens.length) {
      probs = this.targetProbs[this.targetProbs.length - 1];
    } else {
      let adjustedProbs = this.targetProbs[this.verifyIdx!].map((qVal, idx) =>
        Math.max(qVal - this.draftProbs[this.verifyIdx!][idx], 0)
      );
      const sum = adjustedProbs.reduce((acc, val) => acc + val, 0);
      probs = adjustedProbs.map((val) => val / sum);
    }

    const token = sampleFromProbs(probs);
    const decoded = this.tokenizer.decode([token], {
      skip_special_tokens: true,
    });

    this.tokens.push(token);
    console.log(`[worker]: sampled ${decoded} (${token})`);
    self.postMessage({
      type: "update",
      stage: "sample",
      token: decoded,
      promptId: this.currentPromptId!,
    });

    this.state = "draft";
    this.draftTokens = [];
    this.draftProbs = [];
    this.verifyIdx = null;
    return;
  }

  public async step() {
    if (
      !this.inProgress ||
      this.tokens[this.tokens.length - 1] === EOT_TOKEN_ID
    ) {
      return;
    }
    console.log("[worker]: step", this.state);

    if (this.state === "draft") {
      await this.handleDraft();
    } else if (this.state === "verify") {
      await this.handleVerify();
    } else {
      await this.handleSample();
    }

    if (this.tokens[this.tokens.length - 1] === EOT_TOKEN_ID) {
      console.log("[worker]: done");
      this.inProgress = false;
      self.postMessage({
        type: "done",
        promptId: this.currentPromptId!,
      });
    }
  }
}

let worker = new Worker();

self.addEventListener("message", (event: MessageEvent<WorkerMessage>) => {
  const data = event.data;
  if (data.type === "generate") {
    worker.generate(data.prompt, data.promptId).catch((err) => {
      self.postMessage({
        type: "error",
        error: err instanceof Error ? err.message : String(err),
        promptId: data.promptId,
      });
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
    self.postMessage({ type: "ready" });
  })
  .catch((err) => {
    console.error("Failed to initialize models:", err);
    const errorMessage =
      err instanceof Error
        ? `${err.message}${err.stack ? `\n${err.stack}` : ""}`
        : String(err);
    self.postMessage({
      type: "error",
      error: errorMessage,
    });
  });

export {};
