import json
import os

import torch
import math
import copy

from utils import find_optimal_coef

from args import parse_arguments
from eval import (
    evaluate_task_vector,
    evaluate_task_vector_at_coef,
)
from task_vectors import NonLinearTaskVector

###############
#### TSV Merge Orthogonalization
def compute_and_sum_svd_mem_reduction(task_vectors, device):
    sv_reduction = 1 / len(task_vectors)
    print("Computing SVD...")
    with torch.no_grad():
        new_vector_dict = {}
        for key in task_vectors[0].vector:
            new_vector_dict[key] = {}
            for i, task_vector in enumerate(task_vectors):
                vec = task_vector.vector[key].to(device)

                if (
                    len(task_vector.vector[key].shape) == 2
                    and "text_projection" not in key
                ):
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)

                    if i == 0:
                        print(f"Computed SVD for {key}...")
                        sum_u = torch.zeros_like(u, device=device)
                        sum_s = torch.zeros_like(s, device=device)
                        sum_v = torch.zeros_like(v, device=device)
                    reduced_index_s = int(s.shape[0] * sv_reduction)

                    # select only the first reduced_index_s columns of u and place them
                    sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                        :, :reduced_index_s
                    ]
                    sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                        :reduced_index_s
                    ]
                    # select only the first reduced_index_s rows of v and place them
                    sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                        :reduced_index_s, :
                    ]

                else:
                    if i == 0:
                        new_vector_dict[key] = vec.clone()
                    else:
                        new_vector_dict[key] += (vec - new_vector_dict[key]) / (i + 1)

            if len(task_vector.vector[key].shape) == 2 and "text_projection" not in key:
                u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)

                new_vector_dict[key] = torch.linalg.multi_dot(
                    (
                        u_u,
                        v_u,
                        torch.diag(sum_s),
                        u_v,
                        v_v,
                    )
                )
    new_vector = copy.deepcopy(task_vectors[0])
    for key in new_vector_dict:
        new_vector.vector[key] = new_vector_dict[key].to("cpu")
    return new_vector


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

task_vector = compute_and_sum_svd_mem_reduction(task_vectors, "cuda")

args.eval_datasets = [dataset + "Val" for dataset in eval_datasets]
args.control_dataset = None

# We use the validation set to choose the optimal coefficient.
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

save_file = f"{args.save}/additions_tsvm.json"
with open(save_file, "w") as f:
    json.dump(additive_accuracies, f, indent=4)
