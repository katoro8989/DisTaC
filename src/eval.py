import numpy as np
import torch
import tqdm
import torch.nn.functional as F
import utils
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from heads import get_classification_head
from modeling import ImageClassifier

def eval_single_dataset(image_encoder, dataset_name, args):
    classification_head = get_classification_head(args, dataset_name)
    # classification_head = get_classification_head_with_image_encoder(args, image_encoder, dataset_name)
    model = ImageClassifier(image_encoder, classification_head)

    model.eval()
    torch.cuda.set_device(args.device_number)
    args.batch_size = 128

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    device = args.device
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        top1, correct, n = 0.0, 0.0, 0.0
        loss_total = 0.0
        for _, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data["images"].to(device)
            y = data["labels"].to(device)

            logits = utils.get_logits(x, model)

            loss = loss_fn(logits, y)
            loss_total += loss.item()

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()

            n += y.size(0)

        top1 = correct / n
        loss_total /= n

    metrics = {"top1": top1, "loss": loss_total}
    print(f"Done evaluating on {dataset_name}. Accuracy: {100 * top1:.2f}%")

    return metrics

def calculate_entropy_manual(logits: torch.Tensor) -> torch.Tensor:
  probs = F.softmax(logits, dim=-1)

  p_log_p = probs * torch.log(probs.clamp_min(1e-9))
  
  return -torch.sum(p_log_p, dim=-1)


def evaluate(image_encoder, args):
    if args.eval_datasets is None:
        return
    per_dataset_results = {}
    eval_datasets = (
        args.eval_datasets
        if args.control_dataset is None
        else args.eval_datasets + [args.control_dataset]
    )
    for dataset_name in eval_datasets:
        print("Evaluating on", dataset_name)

        results = eval_single_dataset(image_encoder, dataset_name, args)

        print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        per_dataset_results[dataset_name + ":top1"] = results["top1"]

    return per_dataset_results


def evaluate_task_vector_at_coef(
    task_vector, pretrained_checkpoint, args, scaling_coef
):
    image_encoder = task_vector.apply_to(
        pretrained_checkpoint, scaling_coef=scaling_coef
    )

    coef_info = evaluate(image_encoder, args)

    coef_info = add_normalized_accuracy(coef_info, args)
    coef_info["avg_normalized_top1"] = np.mean(
        [coef_info[dataset + ":normalized_top1"] for dataset in args.eval_datasets]
    )
    coef_info["avg_top1"] = np.mean(
        [coef_info[dataset + ":top1"] for dataset in args.eval_datasets]
    )

    return coef_info


def evaluate_task_vector(
    task_vector, pretrained_checkpoint, args
):
    info = {}
    for scaling_coef in np.linspace(0.0, 1.0, args.n_eval_points):
        print(f"Evaluating for scaling coefficient {scaling_coef:.2f}")
        info[scaling_coef] = evaluate_task_vector_at_coef(
            task_vector,
            pretrained_checkpoint,
            args,
            scaling_coef,
        )

    return info


def add_normalized_accuracy(results, args):
    for dataset_name in args.eval_datasets:
        results[dataset_name + ":normalized_top1"] = (
            results[dataset_name + ":top1"] / args.finetuning_accuracies[dataset_name]
        )

    return results
