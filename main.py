import copy
import os
from argparse import ArgumentParser
from types import SimpleNamespace

import textattack
import torch
import transformers
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers import AutoConfig

import wandb
from modelling import bert, roberta
from run_multi_cho import DreamProcessor  # InputFeatures,
from run_multi_cho import (AlphaNliProcessor, HellaswagProcessor, Metrics,
                           ReclorProcessor, SwagProcessor)
from run_sent_clas import MnliProcessor, QqpProcessor, SstProcessor
from tqdm import tqdm
# from ..TextAttack import textattack
mcq = ["alphanli", "dream", "hellaswag", "reclor"]


def convert_examples_to_features_for_multiple_choice(
    examples, label_list, max_seq_length, tokenizer=None
):
    class InputFeatures(object):
        def __init__(self, choices_features, label_id):
            if tokenizer:
                self.choices_features = [
                    {
                        "input_ids": input_ids,
                        "input_mask": input_mask,
                        "segment_ids": segment_ids,
                    }
                    for input_ids, input_mask, segment_ids in choices_features
                ]
            else:
                self.choices_features = choices_features
            self.label_id = label_id

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for i, example in tqdm(enumerate(examples), desc="Converting examples to features"):
        choices_features = []
        for ending_idx, (context, ending) in enumerate(
            zip(example.contexts, example.endings)
        ):
            text_a = context
            if example.question.find("_") != -1:
                # This is for cloze questions.
                text_b = example.question.replace("_", ending)
            else:
                text_b = f"{example.question} {ending}"
            if tokenizer:
                encoded_inputs = tokenizer(
                    text_a,
                    text_b,
                    max_length=max_seq_length,
                    padding="max_length",
                    truncation=True,
                    return_token_type_ids=True,
                )

                choices_features.append(
                    (
                        encoded_inputs["input_ids"],
                        encoded_inputs["attention_mask"],
                        encoded_inputs["token_type_ids"],
                    )
                )
            else:
                choices_features.append((text_a, text_b))

        label_id = label_map[example.label]


        features.append(
            InputFeatures(choices_features=choices_features, label_id=label_id)
        )

    return features


def convert_examples_to_features_for_sequence_classification(
    examples, label_list, max_seq_length, tokenizer=None
):
    class InputFeatures(object):
        def __init__(self, data, label_id):
            if tokenizer is None:
                self.features = data
            else:
                self.input_ids = input_ids
                self.input_mask = input_mask
                self.segment_ids = segment_ids
            self.label_id = label_id

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for i, example in enumerate(examples):
        if tokenizer is not None:
            if example.text_b:
                encoded_inputs = tokenizer(
                    example.text_a,
                    example.text_b,
                    max_length=max_seq_length,
                    padding="max_length",
                    truncation=True,
                    return_token_type_ids=True,
                )
                input_ids = encoded_inputs["input_ids"]
                input_mask = encoded_inputs["attention_mask"]
                segment_ids = encoded_inputs["token_type_ids"]
                # tokens = tokenizer.convert_ids_to_tokens(input_ids)
            else:
                encoded_inputs = tokenizer(
                    example.text_a,
                    max_length=max_seq_length,
                    padding="max_length",
                    truncation=True,
                    return_token_type_ids=True,
                )
                input_ids = encoded_inputs["input_ids"]
                input_mask = encoded_inputs["attention_mask"]
                segment_ids = encoded_inputs["token_type_ids"]
                # tokens = tokenizer.convert_ids_to_tokens(input_ids)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            data = encoded_inputs
        else:
            if example.text_b:
                data = (example.text_a, example.text_b)
            else:
                data = (example.text_a,)

        if len(label_list) == 1:
            label_id = example.label
        else:
            label_id = label_map[example.label]

        features.append(
            InputFeatures(
                data=data,
                label_id=label_id,
            )
        )

    return features


def main(args):
    # load huggingface models
    # model = transformers.AutoModelForSequenceClassification.from_pretrained(
    #     # "textattack/bert-base-uncased-SST-2"
    #     model_path
    # )
    # load custom model
    args.load_model_path = f"model/{args.task_name}_{args.model_type}"
    if args.task_name in mcq:
        if args.model_type == "roberta":
            model = roberta.RobertaForMultipleChoice(config)
        elif args.model_type == "bert":
            model = bert.BertForMultipleChoice(config)
        converter = convert_examples_to_features_for_multiple_choice
    elif args.task_name in ["sst-2", "mnli"]:
        if args.model_type == "roberta":
            model = roberta.RobertaForSequenceClassification(config)
        else:
            model = bert.BertForSequenceClassification(config)
        converter = convert_examples_to_features_for_sequence_classification
    else:
        model = None
        converter = None
    assert model is not None and converter is not None
    model.load_state_dict(
        torch.load(
            os.path.join(args.load_model_path, f"{args.best_epoch}_pytorch_model.bin")
        )
    )
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_and_path)

    recipe_dict = {
        "bertattack": textattack.attack_recipes.BERTAttackLi2020,
        "textfooler": textattack.attack_recipes.TextFoolerJin2019,
    }
    recipe_strat = recipe_dict.get(args.recipe)
    assert recipe_strat, "Please choose a attack recipe"
    task_name = args.task_name.lower()
    processors = {
        "swag": SwagProcessor,
        "dream": DreamProcessor,
        "hellaswag": HellaswagProcessor,
        "alphanli": AlphaNliProcessor,
        "sst-2": SstProcessor,
        "qqp": QqpProcessor,
        "mnli": MnliProcessor,
        "reclor": ReclorProcessor,
    }
    if task_name not in processors:
        raise ValueError(f"Task not found: {task_name}")
    processor = processors[task_name]()
    dataset = processor.get_test_examples(os.path.join(args.data_dir, task_name))
    label_list = processor.get_labels()
    config = AutoConfig.from_pretrained(args.model_and_path)
    
    eval_features = converter(dataset, label_list, args.max_seq_length)
    data = []
    if args.task_name in mcq:
        # generate ctx-choice pairs
        for ev in eval_features:
            list_of_lists = [[*a][1] for a in ev.choices_features]
            flat_list = (
                tuple([ev.choices_features[0][0]] + list_of_lists),
                ev.label_id,
            )
            data.append(flat_list)
        column_name = tuple(["ctx"] + [f"choice{i}" for i in range(len(label_list))])
    else:
        data.extend((ev.features, ev.label_id) for ev in eval_features)
        column_name = ("text", )
    print(f"column_name: {column_name}")
    dataset = textattack.datasets.Dataset(data, input_columns=column_name)
    # built in text attack dataset
    # dataset = textattack.datasets.HuggingFaceDataset(
    #     task_name_model[args.task_name], split="validation" if args.task_name in ["sst-2","reclor"] else "test"
    # )
    
    print(f"model: {os.path.join(args.load_model_path)}")
    # import .textattack
    model_args = SimpleNamespace(
        **{
        }
    )

    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(
        model, tokenizer, model_args
    )

    attack = recipe_strat.build(model_wrapper)
    model_args = textattack.AttackArgs(
        num_examples=-1,
        log_to_csv=os.path.join(
            "data", "asa", f"{args.recipe}_{args.task_name}_log.csv"
        ),
        checkpoint_interval=5,
        checkpoint_dir="checkpoints",
        # parallel=True,
        log_to_wandb={},
        disable_stdout=True,
    )
    attacker = textattack.Attacker(attack, dataset, model_args)
    attacker.attack_dataset()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/glue/",
        help="Directory to contain the input data for all tasks.",
    )
    parser.add_argument(
        "--recipe", type=str, default="bertattack", choices=["textfooler", "bertattack"]
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="Maximum total input sequence length after word-piece tokenization.",
    )
    parser.add_argument("--task_name", default=None)
    parser.add_argument("--model_and_path", default="bert-base-uncased")
    parser.add_argument("--model_type", default="bert")
    parser.add_argument("--best_epoch", default="best")
    args = parser.parse_args()
    wandb.init(project="asa", config=args)
    main(args)
