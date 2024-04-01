from dataclasses import dataclass, field
from typing import Optional, List, Union
from transformers import GenerationConfig
from pathlib import Path


# Goals
GOAL_ARG_GEN = "arg_gen"
GOAL_ANS_IDX = "ans_idx"

# Prompt separators
PROMPT_01_CLEAN = "01_clean"
PROMPT_02_MEDRAG = "02_medrag"

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The local path or huggingface hub name of the model and tokenizer to use."},
    )

    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this"
                " dtype. If `auto` is passed, the dtype will be automatically derived"
                " from the model's weights. We will override this if we use quantization."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    use_lora: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use LoRA. If True, the model will be trained with LoRA: https://arxiv.org/abs/2106.09685"
            )
        },
    )

    quantization: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Whether to use '4' or '8' bit quantization. Requires bitsandbytes library:"
                " https://github.com/TimDettmers/bitsandbytes. This parameter is only used for training."
            )
        },
    )

    quantization_inference: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Whether to use '4' or '8' bit quantization. Requires bitsandbytes library:"
                " https://github.com/TimDettmers/bitsandbytes. This parameter is only used for inference."
            )
        },
    )

    lora_weights_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "If the model has been trained with LoRA, "
                "path or huggingface hub name or local path to the pretrained weights."
            )
        },
    )

    lora_r: Optional[int] = field(
        default=8,
        metadata={"help": "Lora attention dimension."},
    )

    lora_alpha: Optional[float] = field(
        default=16,
        metadata={"help": "The alpha parameter for Lora scaling."},
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": "The dropout probability for Lora layers."},
    )

    lora_target_modules: Optional[List[str]] = field(
        default_factory=list,
        metadata={
            "help": (
                "The target modules to which LoRA will be applied. If not specified, We"
                " will use the default modules for the model in huggingface PEFT library."
            )
        },
    )

    force_auto_device_map: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to force the use of the auto device map. If set to True, the model will be split across "
                "GPUs and CPU to fit the model in memory. If set to False, a full copy of the model will be loaded "
                "into each GPU. Defaults to False."
            )
        },
    )

    use_better_transformer: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to transform the model using Better Transformer library:"
                "https://huggingface.co/docs/optimum/bettertransformer/overview. Requires optimum"
                "'https://huggingface.co/docs/optimum/installation'. Defaults to False. This flag is "
                "only supported for inference, we will override it for training as BetterTransformer does not"
                "support custom attention masks during training."
            )
        },
    )

    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Trust the remote code from HuggingFace model hub. Defaults to False."},
    )

    use_flash_attention: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use FlashAttention. If True, the model will be trained with FlashAttention."
                "Flash attention must be installed, see: https://github.com/Dao-AILab/flash-attention "
                "for more details."
            )
        },
    )

    max_memory_MB: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Free memory per gpu in MB. Used to compute the device map when force_auto_device_map is set to True."
                "Defaults to None."
            )
        },
    )

    merge_lora_after_training: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to merge LoRA layers after training. If True, the model will be trained with LoRA and then"
                " the LoRA layers will be merged into the model. We will save the merged model in"
                " {output_dir}/merged_model. Defaults to False."
            )
        },
    )

    merge_lora_before_inference: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to merge LoRA layers before eval. If True, the model and LoRA layers will be loaded, merged"
                " and saved into {model_name_or_path}/merged_model. Then we will load the merged model using the"
                " provided quantization and torch.dtype values. Note: This flag is only useful is you want to use 4"
                " bit or 8 bit quantization for inference. bf16, fp16 and fp32 models are automatically merged when"
                " loaded for inference, there is no need to save the merged model and reload it. Defaults to False."
            )
        },
    )

    keep_merged_model_after_eval: bool = field(
        default=False,
        metadata={
            "help": (
                "If 'merge_lora_before_inference' is set to True, whether to keep the merged model after eval, or"
                " delete it to save disk space. Merged model will be saved in {model_name_or_path}/merged_model and"
                " will use the same disk space as the original model. While storing only the LoRA layers only takes a"
                " few MBs. Defaults to False."
            )
        },
    )

    do_sample: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use sampling ; use greedy decoding otherwise."
        },
    )

    temperature: float = field(
        default=1.0,
        metadata={"help": "The value used to module the next token probabilities."},
    )

    top_k: int = field(
        default=50,
        metadata={
            "help": (
                "The number of highest probability vocabulary tokens to keep for top-k-filtering."
            )
        },
    )

    top_p: float = field(
        default=1.0,
        metadata={
            "help": (
                "If set to float < 1, only the most probable tokens "
                "with probabilities that add up to top_p or higher are kept for generation."
            )
        },
    )

    repetition_penalty: float = field(
        default=1.0,
        metadata={
            "help": (
                "The parameter for repetition penalty. 1.0 means no penalty. See "
                "https://arxiv.org/abs/1909.05858 for more details."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_files: List[str] = field(
        default=None, metadata={"help": "Input training data files (a jsonl file)."}
    )

    validation_files: List[str] = field(
        default=None,
        metadata={"help": "Input validation data files (a jsonl file)."},
    )

    test_files: List[str] = field(
        default=None, metadata={"help": "Input test data files (a jsonl file)."}
    )

    max_seq_length: int = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences"
                " longer than this will be truncated, sequences shorter will be padded."
            )
        },
    )

    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )

    use_dev_inference: bool = field(
        default=False,
        metadata={"help": "Use the development set for inference instead of the test set."},
    )

    evaluate_all_checkpoints: bool = field(
        default=False,
        metadata={"help": "Evaluate all checkpoints in the model directory."},
    )

    prompt_loss_weight: float = field(
        default=0.05,
        metadata={
            "help": (
                "The weight of the prompt tokens in the loss. If set to '0.05' the prompt tokens will have a total"
                " weight of 5% in the loss while the result tokens will have a total weight of 95%. Only used for"
                " computing the loss in the training data. Defaults to `0.05`."
            )
        },
    )

    prompt_style: str = field(
        default=PROMPT_02_MEDRAG,
        metadata={
            "help": (
                f"Which prompt style to use. Defaults to `{PROMPT_01_CLEAN}`."
                f"It can be '{PROMPT_01_CLEAN}' or '{PROMPT_02_MEDRAG}'."
            )
        },
    )

    grounding_entities: List[str] = field(
        default=None,
        metadata={
            "help": "Entity types to include as grounding."
        },
    )

    grounding_langs: List[str] = field(
        default=None,
        metadata={
            "help": "Grounding source languages to use as grounding. Default uses the language of the example."
        },
    )

    goal: str = field(
        default=None,
        metadata={
            "help": f"Type of goal for this task: generate argument = '{GOAL_ARG_GEN}' or answer classification = '{GOAL_ANS_IDX}'"
        },
    )

    mark_correct: bool = field(
        default=True,
        metadata={
            "help": f"Whether or not we mark the correct option as '(correct)' in the input text."
        },
    )

    max_rag_docs: int = field(
        default=None,
        metadata={
            "help": f"Maximum number of RAG documents to use per example."
        },
    )