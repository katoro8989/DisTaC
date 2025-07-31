import json
import os

from utils import find_optimal_coef

from args import parse_arguments
from eval import (
    evaluate_task_vector,
    evaluate_task_vector_at_coef,
)
from task_vectors import NonLinearTaskVector

args = parse_arguments()


if args.seed is not None:
    args.save = f"{args.save}/checkpoints_{args.seed}/{args.model}"
else:
    args.save = f"{args.save}/checkpoints/{args.model}"


print("*" * 100)
print("Evaluating.")
ft_accuracies_path = os.path.join(args.save, "ft_accuracies.json")
print("*" * 100)

with open(ft_accuracies_path) as f:
    args.finetuning_accuracies = json.load(f)

eval_datasets = [
    "Cars",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "MNIST",
    "RESISC45",
    "SVHN",
    "SUN397",
]

task_vectors = []

for dataset in eval_datasets:
    pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
    finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
    task_vectors.append(
        NonLinearTaskVector(
            model_name=args.model, 
            pretrained_checkpoint=pretrained_checkpoint, 
            finetuned_checkpoint=finetuned_checkpoint
            )
    )

task_vector = sum(task_vectors)

args.eval_datasets = [dataset + "Val" for dataset in eval_datasets]
args.control_dataset = None

val_metrics = evaluate_task_vector(
    task_vector,
    pretrained_checkpoint,
    args,
)

optimal_coef = find_optimal_coef(
    val_metrics,
    metric="avg_normalized_top1",
    minimize=False,
)

# Evaluate on the test set with the optimal coefficient.
args.eval_datasets = [dataset for dataset in eval_datasets]
test_metrics = evaluate_task_vector_at_coef(
    task_vector,
    pretrained_checkpoint,
    args,
    float(optimal_coef),
)

print("=" * 100)
print(f"Test normalized accuracy: {test_metrics['avg_normalized_top1']}")
print(f"Test absolute accuracy: {test_metrics['avg_top1']}")
additive_accuracies = {"test": test_metrics, "val": val_metrics}

save_file = f"{args.save}/additions.json"
with open(save_file, "w") as f:
    json.dump(additive_accuracies, f, indent=4)
