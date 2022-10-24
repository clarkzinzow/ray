import click
import time
import json
import os
import numpy as np
import pandas as pd

from torchvision import transforms
from torchvision.models import resnet18

import ray
from ray.train.torch import TorchCheckpoint, TorchPredictor
from ray.train.batch_predictor import BatchPredictor
from ray.data.preprocessors import BatchMapper


def preprocess(batch: np.ndarray) -> pd.DataFrame:
    """
    User Pytorch code to transform user image.
    """
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return pd.DataFrame({"image": [preprocess(image).numpy() for image in batch]})


def _human_readable_bytes(num_bytes: int) -> str:
    return f"{num_bytes / (2 ** 30)} GiB"


class DebugPredictor(TorchPredictor):
    def call_model(self, tensor):
        import torch

        out = super().call_model(tensor)
        print(
            "cuda reserved memory: ",
            _human_readable_bytes(torch.cuda.memory_reserved()),
        )
        print(
            "cuda allocated memory: ",
            _human_readable_bytes(torch.cuda.memory_allocated()),
        )
        return out


@click.command(help="Run Batch prediction on Pytorch ResNet models.")
@click.option("--data-size-gb", type=int, default=1)
@click.option("--smoke-test", is_flag=True, default=False)
def main(data_size_gb: int, smoke_test: bool = False):
    data_url = (
        f"s3://anonymous@air-example-data-2/{data_size_gb}G-image-data-synthetic-raw"
    )

    if smoke_test:
        # Only read one image
        data_url = [data_url + "/dog.jpg"]
        print("Running smoke test on CPU with a single example")
    else:
        print(
            f"Running GPU batch prediction with {data_size_gb}GB data from {data_url}"
        )

    start = time.time()
    dataset = ray.data.read_images(data_url, size=(256, 256))

    model = resnet18(pretrained=True)

    preprocessor = BatchMapper(preprocess, batch_format="numpy")
    ckpt = TorchCheckpoint.from_model(model=model, preprocessor=preprocessor)

    predictor = BatchPredictor.from_checkpoint(ckpt, DebugPredictor)
    predictor.predict(
        dataset,
        num_gpus_per_worker=int(not smoke_test),
        feature_columns=["image"],
        batch_size=1024,
    )
    total_time_s = round(time.time() - start, 2)

    # For structured output integration with internal tooling
    results = {
        "data_size_gb": data_size_gb,
    }
    results["perf_metrics"] = [
        {
            "perf_metric_name": "total_time_s",
            "perf_metric_value": total_time_s,
            "perf_metric_type": "LATENCY",
        },
        {
            "perf_metric_name": "throughout_MB_s",
            "perf_metric_value": (data_size_gb * 1024 / total_time_s),
            "perf_metric_type": "THROUGHPUT",
        },
    ]

    test_output_json = os.environ.get("TEST_OUTPUT_JSON", "/tmp/release_test_out.json")
    with open(test_output_json, "wt") as f:
        json.dump(results, f)

    print(results)


if __name__ == "__main__":
    main()
