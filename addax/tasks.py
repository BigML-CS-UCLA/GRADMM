import os
import sys
import json
import logging
from dataclasses import dataclass
from typing import Any, List, Union

from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split

from templates import *
from utils import temp_seed


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def read_jsonl(path):
    with open(path, "rb") as fp:
        return [json.loads(x.decode("utf-8")) for x in fp.readlines()]


def get_task(task_name):
    aa = task_name.split("__")
    if len(aa) == 2:
        task_group, subtask = aa
    else:
        task_group = aa[0]
        subtask = None
    class_ = getattr(sys.modules[__name__], f"{task_group}Dataset")
    instance = class_(subtask)
    
    return instance


def get_syn_task(task_name, data_path):
    subtask = None
    class_ = getattr(sys.modules[__name__], f"{task_name}Dataset")
    instance = class_(subtask=subtask, path=data_path)
    
    return instance


def process_task(
    task: Any,
    num_eval_to_keep: int = 100,
    seed: int = 42,
    mix_train_val: bool = False,
    kept_eval_as_train: bool = False,
) -> Any:
    """Process the task by splitting the validation set into test and validation sets.

    Args:
        task: Dataset instance
        num_eval_to_keep: Number of examples to keep in the validation set
        seed: Random seed
        mix_train_val: Whether to mix train and validation sets
        kept_eval_as_train: Whether to use validation set as train set

    Returns:
    """
    samples = task.samples
    train_samples = samples["train"]
    valid_samples = samples["valid"]

    # Split val and test
    # Val set is used for generate synthetic data
    list_labels = [sample.correct_candidate for sample in valid_samples]
    if num_eval_to_keep > 0:
        valid_samples, keep_samples = train_test_split(
            valid_samples,
            stratify=list_labels,
            test_size=num_eval_to_keep,
            random_state=seed,
        )
        # Train on valid set
        if mix_train_val:
            logger.info("Mix train and val sets")
            train_samples += keep_samples
        elif kept_eval_as_train:
            logger.info("Use val set as train set")
            train_samples = keep_samples

    task.samples = {"train": train_samples, "valid": valid_samples}

    return task


@dataclass
class Sample:
    id: int = None
    data: dict = None
    correct_candidate: Union[str, List[str]] = None
    candidates: List[str] = None


class Dataset:
    mixed_set = False
    train_sep = "\n\n"
    generation = False  # whether this is a generation task

    def __init__(self, subtask=None, **kwargs) -> None:
        self.subtask = subtask
        self.samples = None

    def get_task_name(self):
        return self.subtask

    def load_dataset(self):
        raise NotImplementedError

    def get_template(self, template_version=0):
        templates = {0: Template}
        return templates[template_version]

    def build_sample(self):
        return

    def sample_train_sets(
        self,
        num_train=32,
        num_dev=None,
        num_eval=None,
        num_train_sets=None,
        seed=None,
    ):
        if seed is not None:
            # one train/demo set using the designated seed
            seeds = [seed]
        elif num_train_sets is not None:
            # num_train_sets train/demo sets
            seeds = list(range(num_train_sets))
        else:
            # one train/demo set per evaluation sample
            assert num_dev is None  # not supported
            len_valid_samples = (
                len(self.samples["valid"]) if num_eval is None else num_eval
            )
            with temp_seed(0):
                seeds = np.random.randint(0, 10000, len_valid_samples)

        train_samples = []
        for _, set_seed in enumerate(seeds):
            if self.mixed_set:
                raise NotImplementedError
            else:
                if num_dev is not None:
                    train_samples.append(
                        self.sample_subset(
                            data_split="train", seed=set_seed, num=num_train + num_dev
                        )
                    )  # dev set is included at the end of train set
                    if num_train + num_dev > len(self.samples["train"]):
                        logger.warning("num_train + num_dev > available training examples")
                else:
                    train_samples.append(
                        self.sample_subset(
                            data_split="train", seed=set_seed, num=num_train
                        )
                    )
                if num_dev is not None:
                    logger.info(
                        "Sample train set"
                        f" {len(train_samples[-1])}/{len(self.samples['train'])}"
                    )
                    logger.info(f"... including dev set {num_dev} samples")
        return train_samples

    def sample_subset(self, data_split="train", seed=0, num=100, exclude=None):
        with temp_seed(seed):
            samples = self.samples[data_split]
            lens = len(samples)
            index = np.random.permutation(lens).tolist()[
                : num if exclude is None else num + 1
            ]
            if exclude is not None and exclude in index:
                index.remove(exclude)
            else:
                index = index[:num]
        
        return [samples[i] for i in index]

    @property
    def valid_samples(self):
        return self.samples["valid"]


class RottenTomatoesDataset(Dataset):
    train_sep = "\n\n"

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("rotten_tomatoes")
        train_d = d["train"]
        validation_d = d["validation"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    # for generative tasks, candidates are []
    def build_sample(self, example):
        label = int(example["label"])
        return Sample(
            id=None,
            data=example,
            correct_candidate=label,
            candidates=[0, 1],
        )

    def get_template(self, template_version=0):
        return {0: RottenTomatoesTemplate}[template_version]()


class SynRottenTomatoesDataset(Dataset):
    train_sep = "\n\n"

    def __init__(self, subtask=None, path=None, **kwargs) -> None:
        self.load_dataset(path, **kwargs)

    def load_dataset(self, path, **kwargs):
        train_d = load_dataset("json", data_files=path)["train"]
        d = load_dataset("rotten_tomatoes")
        validation_d = d["validation"]

        train_samples = [
            self.build_sample(self.process_sample(example)) for example in train_d
        ]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}
        self.num_train = len(train_samples)
        self.num_valid = len(valid_samples)

    # for generative tasks, candidates are []
    def build_sample(self, example):
        label = int(example["label"])
        return Sample(
            id=None,
            data=example,
            correct_candidate=label,
            candidates=[0, 1],
        )

    def process_sample(self, example):
        if isinstance(example["inputs"], list):
            sentence = example["inputs"][0].split("It was")[0]
        else:
            sentence = example["inputs"].strip()
        example = {
            "text": sentence,
            "label": example["label"],
            "idx": example["id"],
        }

        return example

    def get_template(self, template_version=0):
        return {0: RottenTomatoesTemplate}[template_version]()


class IMDBDataset(Dataset):
    train_sep = "\n\n"

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        train_d = load_dataset(
            "json",
            data_files="../data/imdb/train_len256.jsonl",
        )["train"]
        validation_d = load_dataset(
            "json",
            data_files="../data/imdb/validation_len256.jsonl",
        )["train"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    # for generative tasks, candidates are []
    def build_sample(self, example):
        label = int(example["label"])
        return Sample(
            id=None,
            data=example,
            correct_candidate=label,
            candidates=[0, 1],
        )

    def get_template(self, template_version=0):
        return {0: IMDBTemplate}[template_version]()


class SynIMDBDataset(Dataset):
    train_sep = "\n\n"

    def __init__(self, subtask=None, path=None, **kwargs) -> None:
        self.load_dataset(path, **kwargs)

    def load_dataset(self, path, **kwargs):
        train_d = load_dataset("json", data_files=path)["train"]
        validation_d = load_dataset(
            "json",
            data_files="../data/imdb/validation_len256.jsonl",
        )["train"]

        train_samples = [
            self.build_sample(self.process_sample(example)) for example in train_d
        ]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}
        self.num_train = len(train_samples)
        self.num_valid = len(valid_samples)

    def build_sample(self, example):
        label = int(example["label"])
        return Sample(
            id=None,
            data=example,
            correct_candidate=label,
            candidates=[0, 1],
        )

    def process_sample(self, example):
        if isinstance(example["inputs"], list):
            sentence = example["inputs"][0].split("It was")[0]
        else:
            sentence = example["inputs"].strip()
        example = {
            "text": sentence,
            "label": example["label"],
            "idx": example["id"],
        }

        return example

    def get_template(self, template_version=0):
        return {0: IMDBTemplate}[template_version]()


class RTPolarityDataset(Dataset):
    train_sep = "\n\n"

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        train_d = load_dataset(
            "json",
            data_files="../data/rtpolarity/train.jsonl",
        )["train"]
        validation_d = load_dataset(
            "json",
            data_files="../data/rtpolarityy/validation.jsonl",
        )["train"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    # for generative tasks, candidates are []
    def build_sample(self, example):
        label = int(example["label"])
        return Sample(
            id=None,
            data=example,
            correct_candidate=label,
            candidates=[0, 1],
        )

    def get_template(self, template_version=0):
        return {0: RTPolarityTemplate}[template_version]()


class SynRTPolarityDataset(Dataset):
    train_sep = "\n\n"

    def __init__(self, subtask=None, path=None, **kwargs) -> None:
        self.load_dataset(path, **kwargs)

    def load_dataset(self, path, **kwargs):
        train_d = load_dataset("json", data_files=path)["train"]
        validation_d = load_dataset(
            "json",
            data_files="../data/rtpolarityy/validation.jsonl",
        )["train"]

        train_samples = [
            self.build_sample(self.process_sample(example)) for example in train_d
        ]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}
        self.num_train = len(train_samples)
        self.num_valid = len(valid_samples)

    # for generative tasks, candidates are []
    def build_sample(self, example):
        label = int(example["label"])
        return Sample(
            id=None,
            data=example,
            correct_candidate=label,
            candidates=[0, 1],
        )

    def process_sample(self, example):
        if isinstance(example["inputs"], list):
            sentence = example["inputs"][0].split("It was")[0]
        else:
            sentence = example["inputs"].strip()
        example = {
            "inputs": sentence,
            "label": example["label"],
            "idx": example["id"],
        }

        return example

    def get_template(self, template_version=0):
        return {0: RTPolarityTemplate}[template_version]()


class SST2Dataset(Dataset):
    train_sep = "\n\n"

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("glue", "sst2")
        train_d = d["train"]
        validation_d = d["validation"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    # for generative tasks, candidates are []
    def build_sample(self, example):
        label = int(example["label"])
        return Sample(
            id=example["idx"],
            data=example,
            correct_candidate=label,
            candidates=[0, 1],
        )

    def get_template(self, template_version=0):
        return {0: SST2Template}[template_version]()


class SynSST2Dataset(Dataset):
    train_sep = "\n\n"

    def __init__(self, subtask=None, path=None, **kwargs) -> None:
        self.load_dataset(path, **kwargs)

    def load_dataset(self, path, **kwargs):
        train_d = load_dataset("json", data_files=path)["train"]
        d = load_dataset("glue", "sst2")
        validation_d = d["validation"]

        train_samples = [
            self.build_sample(self.process_sample(example)) for example in train_d
        ]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}
        self.num_train = len(train_samples)
        self.num_valid = len(valid_samples)

    # for generative tasks, candidates are []
    def build_sample(self, example):
        label = int(example["label"])
        return Sample(
            id=example["idx"],
            data=example,
            correct_candidate=label,
            candidates=[0, 1],
        )

    def process_sample(self, example):
        if isinstance(example["inputs"], list):
            sentence = example["inputs"][0].split("It was")[0]
        else:
            sentence = example["inputs"].strip()
        example = {
            "sentence": sentence,
            "label": example["label"],
            "idx": example["id"],
        }

        return example

    def get_template(self, template_version=0):
        return {0: SST2Template}[template_version]()


class COLADataset(Dataset):
  train_sep = "\n\n"

  def __init__(self, subtask=None, **kwargs) -> None:
    self.load_dataset(subtask, **kwargs)

  def load_dataset(self, path, **kwargs):
    d = load_dataset("glue", "cola")
    train_d = d["train"]
    validation_d = d["validation"]
    test_d = d["test"]

    train_samples = [self.build_sample(example) for example in train_d]
    valid_samples = [self.build_sample(example) for example in validation_d]
    test_samples = [self.build_sample(example) for example in test_d]

    self.samples = {
        "train": train_samples,
        "valid": valid_samples,
        "test": test_samples,
    }

  # for generative tasks, candidates are []
  def build_sample(self, example):
    label = int(example["label"])
    return Sample(
        id=example["idx"],
        data=example,
        correct_candidate=label,
        candidates=[0, 1],
    )

  def get_template(self, template_version=0):
    return {0: ColaTemplate}[template_version]()


class CopaDataset(Dataset):
    train_sep = "\n\n"
    mixed_set = False

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        train_examples = load_dataset("super_glue", "copa")["train"]
        valid_examples = load_dataset("super_glue", "copa")["validation"]

        train_samples = [self.build_sample(example) for example in train_examples]
        valid_samples = [self.build_sample(example) for example in valid_examples]
        self.samples = {"train": train_samples, "valid": valid_samples}

    # for generative tasks, candidates are []
    def build_sample(self, example):
        sample = Sample(
            id=example["idx"],
            data=example,
            candidates=[example["choice1"], example["choice2"]],
            correct_candidate=example[f"choice{example['label'] + 1}"],
        )

        return sample

    def get_template(self, template_version=0):
        return {0: CopaTemplate}[template_version]()


class BoolQDataset(Dataset):

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("boolq")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = Sample(
            data=example,
            candidates=["Yes", "No"],
            correct_candidate="Yes" if example["answer"] else "No",
        )

        return sample

    def get_template(self, template_version=2):
        return {0: BoolQTemplate, 1: BoolQTemplateV2, 2: BoolQTemplateV3}[
            template_version
        ]()


class MultiRCDataset(Dataset):

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "multirc")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = Sample(
            data=example, candidates=[0, 1], correct_candidate=example["label"]
        )

        return sample

    def get_template(self, template_version=0):
        return {0: MultiRCTemplate}[template_version]()


class CBDataset(Dataset):

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "cb")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = Sample(
            data=example, candidates=[0, 1, 2], correct_candidate=example["label"]
        )

        return sample

    def get_template(self, template_version=0):
        return {0: CBTemplate}[template_version]()


class WICDataset(Dataset):

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "wic")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = Sample(
            data=example, candidates=[0, 1], correct_candidate=example["label"]
        )

        return sample

    def get_template(self, template_version=0):
        return {0: WICTemplate}[template_version]()


class WSCDataset(Dataset):

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "wsc.fixed")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = Sample(
            data=example, candidates=[0, 1], correct_candidate=example["label"]
        )

        return sample

    def get_template(self, template_version=0):
        return {0: WSCTemplate}[template_version]()


class ReCoRDDataset(Dataset):

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "record")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = Sample(
            data=example,
            candidates=example["entities"],
            correct_candidate=example["answers"],
        )

        return sample

    def get_template(self, template_version=0):
        return {0: ReCoRDTemplateGPT3}[template_version]()


class RTEDataset(Dataset):

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("glue", "rte")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = Sample(
            data=example, candidates=[0, 1], correct_candidate=example["label"]
        )

        return sample

    def get_template(self, template_version=0):
        return {0: RTETemplate}[template_version]()


class TwitterEmotionDataset(Dataset):

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset("dair-ai/emotion", "split")
        train_set = d["train"].filter(lambda x: x["label"] in [0, 1])
        valid_set = d["validation"].filter(lambda x: x["label"] in [0, 1])

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = Sample(
            data=example, candidates=[0, 1], correct_candidate=example["label"]
        )
        return sample

    def get_template(self, template_version=0):
        return {0: TwitterEmotionTemplate}[template_version]()


class SynTwitterEmotionDataset(Dataset):
    train_sep = "\n\n"

    def __init__(self, subtask=None, path=None, **kwargs) -> None:
        self.load_dataset(path, **kwargs)

    def load_dataset(self, path, **kwargs):
        train_d = load_dataset("json", data_files=path)["train"]
        train_d = train_d.rename_column("inputs", "text")
        d = load_dataset("dair-ai/emotion", "split")
        validation_d = d["validation"].filter(lambda x: x["label"] in [0, 1])
        train_samples = [
            self.build_sample(self.process_sample(example)) for example in train_d
        ]
        valid_samples = [self.build_sample(example) for example in validation_d]
        self.samples = {"train": train_samples, "valid": valid_samples}
        self.num_train = len(train_samples)
        self.num_valid = len(valid_samples)

    def process_sample(self, example):
        example["text"] = example["text"].split("Does the tweet express")[0]
        return example

    def build_sample(self, example):
        sample = Sample(
            data=example, candidates=[0, 1], correct_candidate=example["label"]
        )
        return sample

    def get_template(self, template_version=0):
        return {0: TwitterEmotionTemplate}[template_version]()


class SQuADDataset(Dataset):
    metric_name = "f1"
    generation = True

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()

    def load_dataset(self):
        dataset = load_dataset("squad")
        train_examples = dataset["train"]
        valid_examples = dataset["validation"]

        train_samples = [
            self.build_sample(example, idx)
            for idx, example in enumerate(train_examples)
        ]
        valid_samples = [
            self.build_sample(example, idx)
            for idx, example in enumerate(valid_examples)
        ]
        self.samples = {"train": train_samples, "valid": valid_samples}

    # for generative tasks, candidates are []
    def build_sample(self, example, idx):
        answers = example["answers"]["text"]
        assert len(answers) > 0
        return Sample(
            id=idx,
            data={
                "title": example["title"],
                "context": example["context"],
                "question": example["question"],
                "answers": answers,
            },
            candidates=None,
            correct_candidate=answers,
        )

    def get_template(self, template_version=0):
        return {0: SQuADv2Template}[template_version]()


class DROPDataset(Dataset):
    metric_name = "f1"
    generation = True

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()

    def load_dataset(self):
        dataset = load_dataset("drop")
        train_examples = dataset["train"]
        valid_examples = dataset["validation"]

        train_samples = [
            self.build_sample(example, idx)
            for idx, example in enumerate(train_examples)
        ]
        valid_samples = [
            self.build_sample(example, idx)
            for idx, example in enumerate(valid_examples)
        ]
        self.samples = {"train": train_samples, "valid": valid_samples}

    # for generative tasks, candidates are []
    def build_sample(self, example, idx):
        answers = example["answers_spans"]["spans"]
        assert len(answers) > 0
        return Sample(
            id=idx,
            data={
                "context": example["passage"],
                "question": example["question"],
                "answers": answers,
            },
            candidates=None,
            correct_candidate=answers,
        )

    def get_template(self, template_version=0):
        return {0: DROPTemplate}[template_version]()
