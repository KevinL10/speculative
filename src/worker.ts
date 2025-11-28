import {
  AutoModelForCausalLM,
  AutoTokenizer,
  PreTrainedModel,
  PreTrainedTokenizer,
  Tensor,
  softmax,
  type ProgressCallback,
} from "@huggingface/transformers";

const MAX_NEW_TOKENS = 128;

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

type WorkerMessage = { type: "generate"; prompt: string } | { type: "stop" };

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

async function generate(prompt: string) {
  console.log("[worker]: generating for prompt", prompt);
  cancelCurrent?.();
  let cancelled = false;
  cancelCurrent = () => {
    cancelled = true;
  };

  const [tokenizer, model] = await TextGenerationPipeline.getInstance();

  const inputIds = tokenizer.apply_chat_template(
    [{ role: "user", content: prompt }],
    { tokenize: true, return_tensor: false, add_generation_prompt: true }
  );

  if (
    !Array.isArray(inputIds) ||
    !inputIds.every((x) => typeof x === "number")
  ) {
    throw new Error("Tokenization failed");
  }

  let tokens = [...inputIds];
  console.log("[worker]: tokens", tokens);

  self.postMessage({
    type: "generation-start",
  } satisfies WorkerResponse);

  const maxTokens = tokens.length + MAX_NEW_TOKENS;

  while (!cancelled && tokens.length < maxTokens) {
    const input_ids = idsToTensor(tokens);
    const attention_mask = idsToTensor(Array(tokens.length).fill(1));
    const out = await model({
      input_ids,
      attention_mask,
    });

    // console.log("[worker]: out", out.logits[0][out.logits.dims[1] - 1].data);

    const nextProbs = softmax(out.logits[0][out.logits.dims[1] - 1].data);
    const nextToken = sampleFromProbs(nextProbs);
    // console.log("[worker]: nextToken", nextToken);

    tokens = [...tokens, nextToken];

    const decoded = tokenizer.decode([nextToken], {
      skip_special_tokens: true,
    });

    self.postMessage({
      type: "token",
      text: decoded,
    } satisfies WorkerResponse);

    if (nextToken === GEMMA_EOT_TOKEN_ID || tokens.length >= maxTokens) {
      break;
    }
  }

  self.postMessage({ type: "done" } satisfies WorkerResponse);
}

self.addEventListener("message", (event: MessageEvent<WorkerMessage>) => {
  const data = event.data;
  if (data.type === "generate") {
    generate(data.prompt).catch((err) => {
      self.postMessage({
        type: "error",
        error: err instanceof Error ? err.message : String(err),
      } satisfies WorkerResponse);
    });
  } else if (data.type === "stop") {
    cancelCurrent?.();
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
