import json

from args import parse_arguments
from eval import eval_single_dataset
from task_vectors import NonLinearTaskVector

args = parse_arguments()
if args.seed is not None:
    args.save = f"{args.save}/checkpoints_{args.seed}/{args.model}"
else:
    args.save = f"{args.save}/checkpoints/{args.model}"
accuracies = {}


print("*" * 100)
print("Evaluating.")



for dataset in [
    "Cars",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "MNIST",
    "RESISC45",
    "SUN397",
    "SVHN",
]:
    print("*" * 100)
    print(f"Evaluating on {dataset}")

    pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"

    finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"

    try:
        task_vector = NonLinearTaskVector(
            args.model, 
            pretrained_checkpoint=pretrained_checkpoint, 
            finetuned_checkpoint=finetuned_checkpoint
            )
        
    except FileNotFoundError:
        print(f"Error: Could not find {finetuned_checkpoint}.")
        continue


    image_encoder = task_vector.apply_to(
        pretrained_checkpoint, 
        scaling_coef=1.0
        )

    for split in ["val", "test"]:
        # Evaluate
        print("=" * 100)
        print(f"Evaluating on {split} split.")
        eval_dataset = dataset if split == "test" else f"{dataset}Val"

        accuracies[eval_dataset] = eval_single_dataset(
            image_encoder, eval_dataset, args
        )["top1"]

save_path = f"{args.save}/ft_accuracies.json"
with open(save_path, "w") as f:
    json.dump(accuracies, f)
