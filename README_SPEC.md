# Install

1. Install PyTorch first.
2. Install OpenCompass with the following command:

```shell
pip install -e .
# Download dataset to data/ folder
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
```

Detailed installation instructions can be found in the `README.md` file.

# Test Speculative Decoding

## Use Volcengine Ark API as the large model

Try running the minimal speculative decoding script with the following command:

```shell
python3 opencompass/models/speculative_decoding.py \
    --api_key <your_api_key> \
    --local_model <local_model_path> \
    --large_model_name <large_model_name> \
    --prompt 讲一讲正态分布的原理。
```

Replace `<your_api_key>`, `<local_model_path>`, and `<large_model_name>` with your actual values.

Please refer to the official documentation on how to apply for an API key. Volcengine doc: https://www.volcengine.com/docs/82379

## Use vllm as the large model

First, deploy vllm with the following command (using the model name `Qwen3-235B-A22B-Instruct-2507` as an example):

```shell
vllm serve Qwen3-235B-A22B-Instruct-2507 \
      --enable-prefix-caching \
      --tensor-parallel-size 8 \
      --api-key 123 \
```

Then, use this vllm server as the large model with the following command:

```shell
python3 opencompass/models/speculative_decoding.py \
    --api_key 123 \
    --local_model <local_model_path> \
    --base_url http://localhost:8000/v1/chat/completions \
    --large_model_name Qwen3-235B-A22B-Instruct-2507 \
    --prompt 讲一讲正态分布的原理。
```

Replace `<local_model_path>` with your actual local model path.

# Run Evaluation

To run the evaluation script, use the following command:

```shell
opencompass spec_examples/eval_spec_chat.py
```
