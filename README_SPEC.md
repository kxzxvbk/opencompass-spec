# Install

```shell
pip install -e .
# Download dataset to data/ folder
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
```

# Test Speculative Decoding

```shell
python3 opencompass/models/speculative_decoding.py --api_key <your_api_key> --local_model <local_model_path> --large_model_name <large_model_name> --prompt 讲一讲正态分布的原理。
```
