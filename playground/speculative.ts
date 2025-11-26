import {
  AutoTokenizer,
  AutoModelForCausalLM,
  softmax,
  Tensor,
} from "@huggingface/transformers";

function progressCallback(progressInfo: any): void {
  if (progressInfo.status === "progress") {
    console.log(`progress: ${progressInfo.progress}`);
  }
}

function argmax(probs: number[]): number {
  let max = -Infinity;
  let maxIndex = 0;
  for (let i = 0; i < probs.length; i++) {
    if (probs[i] > max) {
      max = probs[i];
      maxIndex = i;
    }
  }
  return maxIndex;
}

function idsToTensor(ids: number[]): Tensor {
  const data = BigInt64Array.from(ids, (x) => BigInt(x));
  return new Tensor("int64", data, [1, ids.length]); // [batch=1, seq_len]
}

// const DRAFT_MODEL = "onnx-community/Qwen2.5-0.5B-Instruct";
// const BASE_MODEL = "onnx-community/Qwen2.5-1.5B-Instruct";
const DRAFT_MODEL = "onnx-community/gemma-3-270m-it-ONNX";
// for faster dev use smaller model for now
// const TARGET_MODEL = DRAFT_MODEL;
const TARGET_MODEL = "onnx-community/gemma-3-1b-it-ONNX";

let tokenizer = await AutoTokenizer.from_pretrained(DRAFT_MODEL);
console.log("loading draft model...");
let draftModel = await AutoModelForCausalLM.from_pretrained(DRAFT_MODEL, {
  progress_callback: progressCallback,
});
console.log("loading target model...");
let targetModel = await AutoModelForCausalLM.from_pretrained(TARGET_MODEL, {
  progress_callback: progressCallback,
});
console.log("target model loaded");

console.log("tokenizing...");
const { input_ids, attention_mask } = await tokenizer("What's your name?");

let tokens: number[] = Array.from(input_ids.data);

const LOOKAHEAD = 10;

while (tokens.length < 100) {
  // LOOP: (see https://arxiv.org/pdf/2302.01318)
  // 1. generate n tokens (and their logits) from the draft model
  const draftTokens: number[] = [];
  const draftProbs: number[][] = [];
  const targetProbs: number[][] = [];

  for (let k = 0; k < LOOKAHEAD; k++) {
    // TODO: figure out how to directly concat tokens without reconstructing the tensor
    const input_ids = idsToTensor([...tokens, ...draftTokens]);
    const attention_mask = idsToTensor(Array(tokens.length + k).fill(1));
    const out = await draftModel({ input_ids, attention_mask });

    const nextProbs = softmax(out.logits[0][out.logits.dims[1] - 1].data);
    const nextToken = argmax(nextProbs);

    draftProbs.push(nextProbs);
    draftTokens.push(nextToken);
  }

  const text = tokenizer.decode([...tokens, ...draftTokens], {
    skip_special_tokens: true,
  });
  console.log("draft-sampled", text);

  // 2. get output logits relative to the target model
  const input_ids = idsToTensor([...tokens, ...draftTokens]);
  const attention_mask = idsToTensor(
    Array(tokens.length + draftTokens.length).fill(1)
  );
  const out = await targetModel({ input_ids, attention_mask });

  for (let t = 0; t < LOOKAHEAD; t++) {
    targetProbs.push(softmax(out.logits[0][tokens.length + t - 1].data));
  }

  let numAccepted = 0;

  for (let t = 0; t < LOOKAHEAD; t++) {
    const r = Math.random();
    const x = draftTokens[t];
    const q = targetProbs[t][x];
    const p = draftProbs[t][x];

    if (r < Math.min(1, q / p)) {
      tokens.push(x);
      numAccepted++;
      console.log("accepted", x);
    } else {
      // sample from adjusted distribution q - p and normalize
      let adjustedProbs = targetProbs[t].map((qVal, idx) =>
        Math.max(qVal - draftProbs[t][idx], 0)
      );
      const sum = adjustedProbs.reduce((acc, val) => acc + val, 0);
      adjustedProbs = adjustedProbs.map((val) => val / sum);

      const adjustedToken = argmax(adjustedProbs);
      tokens.push(adjustedToken);
      numAccepted++;
      break;
    }
  }

  const targetText = tokenizer.decode([...tokens], {
    skip_special_tokens: true,
  });
  console.log("target-sampled", targetText);

  // sample one more token from the target model if all tokens are accepted
  if (numAccepted === LOOKAHEAD) {
    const input_ids = idsToTensor([...tokens]);
    const attention_mask = idsToTensor(Array(tokens.length).fill(1));
    const out = await targetModel({ input_ids, attention_mask });
    const nextProbs = softmax(out.logits[0][out.logits.dims[1] - 1].data);
    const nextToken = argmax(nextProbs);
    tokens.push(nextToken);
  }

  if (Array.from(tokens).includes(tokenizer.eos_token_id)) {
    console.log("EOS token found, stopping generation.");
    break;
  }
}
