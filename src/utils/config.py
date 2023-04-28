import os
import json
from typing import Optional
from dataclasses import dataclass, field

@dataclass
class DatasetAttr:
    load_from: str
    dataset_name: Optional[str] = None
    file_name: Optional[str] = None
    file_sha1: Optional[str] = None

    def __post_init__(self):
        self.prompt_column = "instruction"
        self.query_column = "input"
        self.response_column = "output"
        self.history_column = None


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name."}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co."}
    )
    use_fast_tokenizer: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}
    )
    model_revision: Optional[str] = field(
        default=None,
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."}
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Will use the token generated when running `huggingface-cli login`."}
    )
    resize_position_embeddings: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to resize the position embeddings if `max_source_length` exceeds or not."}
    )
    quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the model."}
    )
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory containing the model checkpoints as well as the configurations."}
    )

    def __post_init__(self):
        if self.checkpoint_dir is not None:  # support merging lora weights
            self.checkpoint_dir = [cd.strip() for cd in self.checkpoint_dir.split(",")]


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    dataset: Optional[str] = field(
        default="alpaca_zh",
        metadata={"help": "The name of provided dataset(s) to use. Use comma to separate multiple datasets."}
    )
    dataset_dir: Optional[str] = field(
        default="/root/autodl-tmp/chatglm_efficient_tune/data/alpaca_data",
        metadata={"help": "The name of the folder containing datasets."}
    )
    split: Optional[str] = field(
        default="train",
        metadata={"help": "Which dataset split to use for training and evaluation."}
    )
    overwrite_cache: Optional[bool] = field(
        default=True,
        metadata={"help": "Overwrite the cached training and evaluation sets."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."}
    )
    max_source_length: Optional[int] = field(
        default=768,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    max_target_length: Optional[int] = field(
        default=768,
        metadata={"help": "The maximum total output sequence length after tokenization."}
    )
    max_train_samples: Optional[int] = field(
        default=10000,
        metadata={"help": "For debugging purposes, truncate the number of training examples for each dataset."}
    )
    max_eval_samples: Optional[int] = field(
        default=10000,
        metadata={"help": "For debugging purposes, truncate the number of evaluation examples for each dataset."}
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={"help": "Number of beams to use for evaluation. This argument will be passed to `model.generate`"}
    )
    ignore_pad_token_for_loss: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."}
    )
    source_prefix: Optional[str] = field(
        default=None,
        metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    def __post_init__(self):  # support mixing multiple datasets
        dataset_names = [ds.strip() for ds in self.dataset.split(",")]
        dataset_info = json.load(open(os.path.join(self.dataset_dir, "dataset_info.json"), "r"))

        self.dataset_list = []
        for name in dataset_names:
            if name not in dataset_info:
                raise ValueError("Undefined dataset {} in dataset_info.json.".format(name))

            if "hf_hub_url" in dataset_info[name]:
                dataset_attr = DatasetAttr("hf_hub", dataset_name=dataset_info[name]["hf_hub_url"])
            elif "script_url" in dataset_info[name]:
                dataset_attr = DatasetAttr("script", dataset_name=dataset_info[name]["script_url"])
            else:
                dataset_attr = DatasetAttr(
                    "file",
                    file_name=dataset_info[name]["file_name"],
                    file_sha1=dataset_info[name]["file_sha1"] if "file_sha1" in dataset_info[name] else None
                )

            if "columns" in dataset_info[name]:
                dataset_attr.prompt_column = dataset_info[name]["columns"]["prompt"]
                dataset_attr.query_column = dataset_info[name]["columns"]["query"]
                dataset_attr.response_column = dataset_info[name]["columns"]["response"]
                dataset_attr.history_column = dataset_info[name]["columns"]["history"]

            self.dataset_list.append(dataset_attr)


@dataclass
class FinetuningArguments:
    """
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """
    finetuning_type: Optional[str] = field(
        default="lora",
        metadata={"help": "The name of fine-tuning technique."}
    )
    num_layer_trainable: Optional[int] = field(
        default=3,
        metadata={"help": "Number of trainable layers for Freeze fine-tuning."}
    )
    pre_seq_len: Optional[int] = field(
        default=256,
        metadata={"help": "Number of prefix tokens to use for P-tuning v2."}
    )
    prefix_projection: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to add a project layer for the prefix in P-tuning v2 or not."}
    )
    lora_rank: Optional[int] = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."}
    )
    lora_alpha: Optional[float] = field(
        default=32.0,
        metadata={"help": "The scale factor for LoRA fine-tuning. (similar with the learning rate)"}
    )
    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."}
    )
    lora_target: Optional[str] = field(
        default="query_key_value",
        metadata={"help": "The name(s) of target modules to apply LoRA. Use comma to separate multiple modules."}
    )
    resume_lora_training: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to resume training from the last LoRA weights or create new weights after merging them."}
    )
    plot_loss: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to plot the training loss after fine-tuning or not."}
    )

    def __post_init__(self):
        self.lora_target = [target.strip() for target in
                            self.lora_target.split(",")]  # support custom target modules of LoRA

        if self.num_layer_trainable > 0:  # fine-tuning the last n layers if num_layer_trainable > 0
            self.trainable_layers = ["layers.{:d}.mlp".format(27 - k) for k in range(self.num_layer_trainable)]
        else:  # fine-tuning the first n layers if num_layer_trainable < 0
            self.trainable_layers = ["layers.{:d}.mlp".format(k) for k in range(-self.num_layer_trainable)]

        if self.finetuning_type not in ["none", "freeze", "p_tuning", "lora"]:
            raise NotImplementedError("Invalid fine-tuning method.")
