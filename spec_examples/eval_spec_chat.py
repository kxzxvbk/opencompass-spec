from mmengine.config import read_base

from opencompass.models.speculative_decoding import SpecModel
from opencompass.openicl import ChatInferencer
from opencompass.partitioners import SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import \
        gsm8k_datasets as datasets

models = [
    dict(
        abbr='spec-model',                       # Name of this experiment.
        type=SpecModel,                          # Model class.
        path='path/to/local/model',              # Path to the local model.
        api_key='<your_api_key>',                # API key for the large model.
        large_model_name='<large_model_name>',   # Name of the large model.
        draft_length=32,                         # Draft length for speculative decoding.
        use_spec=True,                           # Whether to use speculative decoding.
        batch_size=1,                            # Batch size for inference. Only support 1.
        run_cfg=dict(num_gpus=8, num_procs=8),
    )
]

for dataset in datasets:
    # Use ChatInferencer instead of GenInferencer
    dataset['infer_cfg']['inferencer'] = dict(type=ChatInferencer)

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000),
    runner=dict(type=LocalRunner,
                max_num_workers=16,
                task=dict(type=OpenICLInferTask)),
)
