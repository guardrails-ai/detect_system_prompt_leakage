from typing import TypedDict
from datasets import load_dataset
from guardrails import Guard, OnFailAction
from validator import DetectSystemPromptLeakage
from rich import print

class Stats(TypedDict):
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    accuracy: float
    false_positive_rate: float
    false_negative_rate: float
    precision: float
    recall: float
    f1: float

def run_stats(prelim_stats: Stats, dataset_len: int):
    accuracy = ((dataset_len - prelim_stats["false_positives"] - prelim_stats["false_negatives"]) / dataset_len) * 100
    false_positive_rate = (prelim_stats["false_positives"] / dataset_len) * 100
    false_negative_rate = (prelim_stats["false_negatives"] / dataset_len) * 100
    
    precision = (
        prelim_stats["true_positives"] / (prelim_stats["true_positives"] + prelim_stats["false_positives"])
        if prelim_stats["true_positives"] > 0 or prelim_stats["false_positives"] > 0
        else 0
    )
    recall = (
        prelim_stats["true_positives"] / (prelim_stats["true_positives"] + prelim_stats["false_negatives"])
        if prelim_stats["true_positives"] > 0 or prelim_stats["false_negatives"] > 0
        else 0 
    )
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if precision > 0 or recall > 0
        else 0
    )

    prelim_stats.update({
        "accuracy": accuracy,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    return prelim_stats

results: Stats = {
    "true_positives": 0,
    "false_positives": 0,
    "true_negatives": 0,
    "false_negatives": 0,
    "accuracy": 0,
    "false_positive_rate": 0,
    "false_negative_rate": 0,
    "precision": 0,
    "recall": 0,
    "f1": 0
}

ds = load_dataset("gabrielchua/system-prompt-leakage")
dataset = ds["test"]

## To run a smaller subset of the dataset
# dataset_size = min(1000, len(test))
# dataset = test.shuffle(seed=42).select(range(dataset_size))

row_index = 0
for row in dataset:
    print(f"Benchmarking against row {row_index}")
    system_prompt: str = row["system_prompt"]  # type: ignore
    content: str = row["content"]  # type: ignore
    should_fail: bool = row["leakage"] == 1  # type: ignore

    guard = Guard().use(
        DetectSystemPromptLeakage(
            system_prompt=system_prompt,
            on_fail=OnFailAction.NOOP
        )
    )

    res = guard.parse(content)
    did_fail = not res.validation_passed
    tp = did_fail and should_fail
    fp = did_fail and not should_fail
    tn = not did_fail and not should_fail
    _fn = not did_fail and should_fail
    if tp:
        results["true_positives"] += 1
    elif fp:
        results["false_positives"] += 1
    elif tn:
        results["true_negatives"] += 1
    elif _fn:
        results["false_negatives"] += 1

    row_index += 1

results = run_stats(results, len(dataset))

print(results)

"""
With default settings
{
    'true_positives': 30317,
    'false_positives': 3134,
    'true_negatives': 37279,
    'false_negatives': 621,
    'accuracy': 94.73728469117462,
    'false_positive_rate': 4.392370113943743,
    'false_negative_rate': 0.8703451948816415,
    'precision': 0.9063107231472901,
    'recall': 0.9799275971297433,
    'f1': 0.9416825855347963
}
"""