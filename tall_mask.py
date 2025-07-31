import torch
from typing import List, Optional
import copy
import os
import numpy as np
from collections import OrderedDict


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
            sorted_reference_dict[key] = sorted_reference_dict["transformer.shared.weight"]
    return sorted_reference_dict


def state_dict_to_vector(state_dict, remove_keys=[]):
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector([value.reshape(-1) for key, value in sorted_shared_state_dict.items()])

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
        tall_masks[tall_mask_lambda] = {key: value for key, value in zip(config.DATASETS, masks_at_scale)}
    return tall_masks


def load_tall_mask(remove_keys, ptm_check, config):
    """Loads TALL masks from disk, unpack and transform to state dictionaries."""
    mask_location = config.model_location.replace("checkpoints", "tall_masks")
    try:
        if config.method.use_ties:
            print("==== Loading TALL Masks built with TIES ====")
            tall_masks = torch.load(
                os.path.join(
                    mask_location,
                    config.model,
                    f"TALL_mask_{config.num_tasks}task_use_ties.npy",
                )
            )
        else:
            print("==== Loading TALL Masks built with Task Arithmetic ====")
            tall_masks = torch.load(os.path.join(mask_location, config.model, f"TALL_mask_{config.num_tasks}task.npy"))
    except:
        raise Exception("TALL Masks are not constructed yet.")

    # unpack masks and convert back to torch tensors
    tall_masks = {k: torch.from_numpy(np.unpackbits(v)) for k, v in tall_masks.items()}

    # convert vectors to dictionaries
    tall_masks = {
        dataset: vector_to_state_dict(mask, ptm_check, remove_keys=remove_keys) for dataset, mask in tall_masks.items()
    }

    return tall_masks

def construct_consensus_mask(ptm_check, prun_thre_k, config, remove_keys=[]):
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
    tall_masks = load_tall_mask(remove_keys, ptm_check, config)
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