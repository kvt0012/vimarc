import glob
import logging
import os
import re
import shutil
from allennlp.commands.train import train_model_from_file

logger = logging.getLogger(__name__)


def train_fixture(config_prefix: str) -> None:
    import vimarc # noqa F401: Needed to register the registrables.
    config_file = config_prefix + "experiment.json"
    if not os.path.exists(config_file):
        config_file = config_prefix + "experiment.jsonnet"

    serialization_dir = config_prefix + "serialization"
    # Train model doesn't like it if we have incomplete serialization
    # directories, so remove them if they exist.
    if os.path.exists(serialization_dir):
        shutil.rmtree(serialization_dir)

    # train the model
    train_model_from_file(config_file, serialization_dir)

    # remove unnecessary files
    shutil.rmtree(os.path.join(serialization_dir, "log"))

    for filename in glob.glob(os.path.join(serialization_dir, "*")):
        if (
            filename.endswith(".log")
            or filename.endswith(".json")
            or re.search(r"epoch_[0-9]+\.th$", filename)
        ):
            os.remove(filename)


if __name__ == "__main__":
    models = [
        "numnet"
    ]
    for model in models:
        train_fixture(f"fixtures/{model}/")
