import json
import os

import torch
import copy
import numpy as np
import time
from utils import find_optimal_coef

from collections import OrderedDict

from args import parse_arguments
from image import eval
from eval import (
    evaluate_task_vector_at_coef,
    add_normalized_accuracy,
    eval_single_dataset,
)
from task_vectors import NonLinearTaskVector

from typing import List, Optional

def state_dict_to_vector(state_dict, remove_keys=[]):
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )


def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    # create a shared state dict using the refence dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    # add back the encoder and decoder embedding weights.
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict[
                "transformer.shared.weight"
            ]
    return sorted_reference_dict

def generate_task_masks(
    tv_flat_checks: torch.Tensor,
    flat_ft: torch.Tensor,
    flat_ptm: torch.Tensor,
    tv: Optional[torch.Tensor] = None,
    tall_mask_lambda: float = 1.0,
) -> torch.Tensor:
    """
    Generate task-specific TALL masks
    TALL masks are generated as: mask_t = |theta_0 - theta_t| > |theta_mt - theta_t| * lambda

    Args:
        tv_flat_checks: individual task vectors
        flat_ft: individual theta_t (fine-tuned weights)
        flat_ptm: theta_0 (pre-trained weight)
        tv: multi-task vector
        tall_mask_lambda: hyper-parameter lambda for generating TALL masks
    Returns:
        final_mask: generated TALL masks with the given lambda, in shape (n_task, n_parameter)
    """

    print(f"Generating TALL masks.")

    if tv is None:
        tv = tv_flat_checks.sum(0)

    flat_multi = flat_ptm + tv

    original_shape = flat_ft.shape

    # generate masks by comparing the l1 distance between |theta_0 - theta_t| and |theta_mt - theta_t|
    diff_pt_ft = (flat_ptm - flat_ft).abs()
    diff_multi_ft = (flat_multi - flat_ft).abs()
    # compare the l1 distance, scaled with hyper-parameter lambda
    mask = diff_pt_ft > diff_multi_ft * tall_mask_lambda

    final_mask = mask.squeeze() if original_shape == tv_flat_checks.squeeze().shape else mask

    print(
        f"Average sparsity for the mask with tall_mask_lambda of {tall_mask_lambda}: {final_mask.float().mean():.4f}"
    )

    return final_mask


def construct_tall_mask(
    tv_flat_checks: torch.Tensor,
    flat_ft: torch.Tensor,
    flat_ptm: torch.Tensor,
    merged_tv: torch.Tensor,
    ptm_check: torch.Tensor,
    remove_keys: List[str],
    config,
):
    """
    Construct TALL masks for all tasks for each lambda, and store in dictionary

    Args:
        tv_flat_checks: individual task vectors
        flat_ft: individual theta_t (fine-tuned weights)
        flat_ptm: theta_0 (pre-trained weight)
        merged_tv: multi-task vector
        ptm_check: pre-trained weight as state dictionary
        remove_keys: the keys to be removed when converting between dictionary and vector
    Returns:
        tall_masks: constructed TALL masks in dictionary format of {lambda: {task: mask}}
    """
    tall_masks = {}
    for tall_mask_lambda in [0.2, 0.3, 0.4, 0.5, 0.6]:
        # generate tall masks for each lambda
        masks_at_scale = generate_task_masks(
            tv_flat_checks, flat_ft, flat_ptm, tall_mask_lambda=tall_mask_lambda, tv=merged_tv
        )
        # convert vectors to dictionary
        masks_at_scale = [vector_to_state_dict(mask, ptm_check, remove_keys=remove_keys) for mask in masks_at_scale]
        # store the masks with {dataset: mask}
        tall_masks[tall_mask_lambda] = {key: value for key, value in zip(config.eval_datasets, masks_at_scale)}
    return tall_masks


def find_optimal_mask(val_metrics, eval_masks, args, time_stamp, save_masks=True):
    """
    Respectively finds the optimal mask for each data task based on the validation accuracy

    Args:
        val_metrics: validation metrics for each lambda
        eval_masks: all generated masks

    Returns:
        best_masks_for_test: the best masks for each task, selected based on validation accuracy from each task
        best_val_metrics: best validation metrics for each task
    """
    # transpose the dict from lambda-task to task-lambda
    transposed_dict = {}
    for key, inner_dict in val_metrics.items():
        for inner_key, value in inner_dict.items():
            if inner_key not in transposed_dict:
                transposed_dict[inner_key] = {}
            transposed_dict[inner_key][key] = value

    # for each task, find the best lambda
    max_subkeys = {key: max(inner_dict, key=inner_dict.get) for key, inner_dict in transposed_dict.items()}

    # select the best mask for each task, which will be used for testing later
    best_masks_for_test = {}
    best_masks_for_test_vector = {}  # the selected masks as vectors
    best_val_metrics = {}
    # respectively for each task:
    for ds in args.eval_datasets:
        # select the lambda which achieves the best valdiation accuracy
        best_lambda = float(max_subkeys[ds + "Val:top1"])
        # select the mask based on the selected lambda, save as dictionaries
        best_masks_for_test[ds] = eval_masks[best_lambda][ds]
        # select the mask based on the selected lambda, save as vectors
        best_masks_for_test_vector[ds] = state_dict_to_vector(eval_masks[best_lambda][ds], remove_keys=[])
        print(f"Best lambda for {ds} is {best_lambda}")
        # save the best validation metric based on the selected lambda
        best_val_metrics[ds + "Val:top1"] = val_metrics[best_lambda][ds + "Val:top1"]

    # save the best masks in disk
    if save_masks:
        # convert to numpy to save with np.packbits for saving storage
        best_masks_for_test_vector = {k: np.packbits(v) for k, v in best_masks_for_test_vector.items()}
        mask_save_dir = os.path.join(args.save, "tall_masks")
        mask_name = (
            f"TALL_mask_{args.num_tasks}task_{time_stamp}.npy"
        )
        if not os.path.exists(os.path.join(mask_save_dir, args.model)):
            os.makedirs(os.path.join(mask_save_dir, args.model))
        np.save(os.path.join(mask_save_dir, args.model, mask_name), best_masks_for_test_vector)
        del best_masks_for_test_vector

    return best_masks_for_test, best_val_metrics


def load_tall_mask(remove_keys, ptm_check, config, time_stamp):
    """Loads TALL masks from disk, unpack and transform to state dictionaries."""
    mask_location = os.path.join(config.save, "tall_masks")
    try:
        tall_masks = np.load(os.path.join(mask_location, config.model, f"TALL_mask_{config.num_tasks}task_{time_stamp}.npy"), allow_pickle=True).item()
    except:
        raise Exception("TALL Masks are not constructed yet.")

    # unpack masks and convert back to torch tensors
    tall_masks = {k: torch.from_numpy(np.unpackbits(v)) for k, v in tall_masks.items()}

    # convert vectors to dictionaries
    tall_masks = {
        dataset: vector_to_state_dict(mask, ptm_check, remove_keys=remove_keys) for dataset, mask in tall_masks.items()
    }

    return tall_masks


def construct_consensus_mask(ptm_check, prun_thre_k, config, time_stamp, remove_keys=[]):
    """
    Generate consensus mask by filtering out least-used parameters

    Args:
        ptm_check: pretrained_checkpoint as state dictionary
        prun_thre_k: weight-pruning threhold, stands for the least number of activated tasks for a parameter to be preserved from pruning
                if prun_thre_k is set to 2: remove both catastrophic and selfish weights;
                if prun_thre_k is set to 1: remove only catastrophic weights;
                if prun_thre_k is set to 0: remove no weights -> reduce to TA or TIES
                if prun_thre_k is set to > num_tasks: remove all weights -> reduce to zero-shot
    Returns:
        consensus_mask_vector: constructed consensus mask as vector (boolean in shape (n_parameter, ))
    """

    print("==== Generating Consensus Mask ====")
    # load TALL masks (in shape (n_task, n_parameter))
    tall_masks = load_tall_mask(remove_keys, ptm_check, config, time_stamp)
    tall_masks = list(tall_masks.values())

    # generate consensus masks
    consensus_mask = copy.deepcopy(tall_masks[0])
    for key, value in consensus_mask.items():
        consensus_mask[key] = torch.zeros_like(value)
        # count for each parameter, the tasks it has been activated for
        for mask in tall_masks:
            consensus_mask[key] = consensus_mask[key] + mask[key].float()
        # filter out the least-activated parameters based on given threshold
        consensus_mask[key] = consensus_mask[key].float() >= prun_thre_k
    consensus_mask_vector = state_dict_to_vector(consensus_mask, remove_keys=remove_keys)

    return consensus_mask_vector

def evaluate(task_vector, pretrained_checkpoint, args, eval_masks=None):
    if args.eval_datasets is None:
        return
    per_dataset_results = {}
    eval_datasets = (
        args.eval_datasets
    )
    for dataset_name in eval_datasets:
        print("Evaluating on", dataset_name)

        sparse_task_vector = copy.deepcopy(task_vector)
        # remove "Val" from dataset_name
        mask = (
            eval_masks[dataset_name[:-3]]
            if "Val" in dataset_name
            else eval_masks[dataset_name]
        )
        # apply mask to sparsify the task vectors with Hadamard product
        for k in mask.keys():
            sparse_task_vector.vector[k] = sparse_task_vector.vector[k] * mask[k].bool().cpu()
        
        image_encoder = sparse_task_vector.apply_to(
            pretrained_checkpoint, scaling_coef=1.0
        )

        results = eval_single_dataset(image_encoder, dataset_name, args)

        print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        per_dataset_results[dataset_name + ":top1"] = results["top1"]

    return per_dataset_results


def evaluate_task_vector_at_coef(
    task_vector, pretrained_checkpoint, args, eval_masks=None
):
    coef_info = evaluate(task_vector, pretrained_checkpoint, args, eval_masks)

    coef_info = add_normalized_accuracy(coef_info, args)
    coef_info["avg_normalized_top1"] = np.mean(
        [coef_info[dataset + ":normalized_top1"] for dataset in args.eval_datasets]
    )
    coef_info["avg_top1"] = np.mean(
        [coef_info[dataset + ":top1"] for dataset in args.eval_datasets]
    )

    return coef_info

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

time_stamp = time.strftime("%Y%m%d-%H%M%S")

ft_dict = []

for dataset in eval_datasets:
    pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
    finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"

    finetuned_dict = torch.load(finetuned_checkpoint, map_location="cpu")

    finetuned_dict = {f"model.{k}": v for k, v in finetuned_dict.items()}

    ft_dict.append(finetuned_dict)

task_vector = NonLinearTaskVector(
    model_name=args.model, 
    pretrained_checkpoint=pretrained_checkpoint, 
    finetuned_checkpoint=finetuned_checkpoint
    )

for dict in ft_dict:
    for key in task_vector.vector:
        if key not in dict:
            dict[key] = task_vector.vector[key]

flat_ft = torch.vstack([state_dict_to_vector(dict, remove_keys=[]) for dict in ft_dict])
ptm_check = torch.load(pretrained_checkpoint, map_location="cpu")
ptm_check = {f"model.{k}": v for k, v in ptm_check.items()}
for key in task_vector.vector:
    if key not in ptm_check:
        ptm_check[key] = task_vector.vector[key]

flat_ptm = torch.vstack([state_dict_to_vector(ptm_check, remove_keys=[])])

tv_flat_checks = flat_ft - flat_ptm

merged_tv = tv_flat_checks.sum(0)

merged_tv_dict = vector_to_state_dict(merged_tv, ptm_check, remove_keys=[])
for key in merged_tv_dict:
    print(key)
    task_vector.vector[key] = merged_tv_dict[key]

args.num_tasks = len(eval_datasets)
args.eval_datasets = eval_datasets


eval_masks = construct_tall_mask(
    tv_flat_checks,
    flat_ft,
    flat_ptm,
    merged_tv,
    ptm_check,
    [],
    args,
    )

args.eval_datasets = [dataset + "Val" for dataset in eval_datasets]
val_metrics = {}
for tall_mask_lambda in [0.2, 0.3, 0.4, 0.5, 0.6]:
    val_metrics[tall_mask_lambda] = evaluate_task_vector_at_coef(
        task_vector,
        pretrained_checkpoint,
        args,
        eval_masks[tall_mask_lambda],
        )

args.eval_datasets = eval_datasets
best_masks_for_test, best_val_metrics = find_optimal_mask(
    val_metrics, eval_masks, args, time_stamp, save_masks=True
)

consensus_mask = construct_consensus_mask(
    ptm_check, 2, args, time_stamp, []
)
merged_tv = merged_tv * consensus_mask
merged_tv_dict = vector_to_state_dict(merged_tv, ptm_check, remove_keys=[])

for key in merged_tv_dict:
    task_vector.vector[key] = merged_tv_dict[key]

args.eval_datasets = [dataset + "Val" for dataset in eval_datasets]
args.control_dataset = None

# We use the validation set to choose the optimal coefficient.
val_metrics = eval.evaluate_task_vector(
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
test_metrics = eval.evaluate_task_vector_at_coef(
    task_vector,
    pretrained_checkpoint,
    args,
    float(optimal_coef),
)
print("=" * 100)
print(f"Test normalized accuracy: {test_metrics['avg_normalized_top1']}")
print(f"Test absolute accuracy: {test_metrics['avg_top1']}")
additive_accuracies = {"test": test_metrics, "val": val_metrics}

save_file = f"{args.save}/additions_consensus.json"
with open(save_file, "w") as f:
    json.dump(additive_accuracies, f, indent=4)
