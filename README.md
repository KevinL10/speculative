# Speculative

Visualized speculative decoding in the browser! Try it out here: https://speculative.vercel.app/.

![speculative decoding](./assets/speculative-decoding.gif)

The default uses `gemma-3-270m-it` as the draft model and `gemma-3-1b-it-ONNX-GQA` as the target model. Inference is done with [transformers.js](https://github.com/huggingface/transformers.js) on WebGPU.
