import os
import time

import torch
import wandb
import numpy as np
import random

from args import parse_arguments
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from distributed import (
    cleanup_ddp,
    distribute_loader,
    is_main_process,
    setup_ddp,
)
from eval import eval_single_dataset
from heads import get_classification_head
from modeling import ImageClassifier, ImageEncoder
from utils import LabelSmoothing, cosine_lr
from optimizer import build_optimizer


def finetune(rank, args, group):
    setup_ddp(rank, args.world_size, port=args.port)

    run = wandb.init(
        config=vars(args),
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        name=f"process_{rank}",
        group=group,
    )

    train_dataset = args.train_dataset
    ckpdir = os.path.join(args.save, train_dataset)

    # Check if checkpoints already exist
    ft_path = os.path.join(args.save, train_dataset, "finetuned.pt")
    zs_path = os.path.join(args.save, train_dataset, "zeroshot.pt")

    assert train_dataset is not None, "Please provide a training dataset."

    if args.load is not None and args.load.endswith("pt") and os.path.exists(args.load):
        aug_cfg = {
            "use_timm": args.use_timm_aug,
        }
        image_encoder = ImageEncoder.load(args.model, args.load, aug_cfg=aug_cfg)
    else:
        print("Building image encoder.")
        image_encoder = ImageEncoder(args)

    classification_head = get_classification_head(args, train_dataset)

    model = ImageClassifier(image_encoder, classification_head)

    model.freeze_head()
    model = model.cuda()

    preprocess_fn = model.train_preprocess
    print_every = 100

    dataset = get_dataset(
        train_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    data_loader = get_dataloader(
        dataset, is_train=True, args=args, image_encoder=None)
    num_batches = len(dataset.train_loader)

    # Distribute the data and model across the GPUs.
    ddp_loader = distribute_loader(data_loader)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        find_unused_parameters=True,
        output_device=rank,
    )

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = build_optimizer(args, params)

    scheduler = cosine_lr(
        optimizer,
        args.lr,
        args.warmup_length,
        args.epochs * num_batches // args.num_grad_accumulation,
    )

    # Saving zero-shot model
    if args.save is not None and is_main_process():
        os.makedirs(ckpdir, exist_ok=True)
        model_path = os.path.join(ckpdir, "zeroshot.pt")
        ddp_model.module.image_encoder.save(model_path)

    for epoch in range(args.epochs):
        ddp_model.train()

        for i, batch in enumerate(ddp_loader):
            start_time = time.time()

            step = (
                i // args.num_grad_accumulation
                + epoch * num_batches // args.num_grad_accumulation
            )

            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()
            labels = batch["labels"].cuda()
            data_time = time.time() - start_time

            logits = ddp_model(inputs)

            loss = loss_fn(logits, labels)

            loss.backward()

            if (i + 1) % args.num_grad_accumulation == 0:
                scheduler(step)

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            batch_time = time.time() - start_time

            if (
                args.checkpoint_every > 0
                and step % args.checkpoint_every == 0
                and is_main_process()
            ):
                print("Saving checkpoint.")
                model_path = os.path.join(ckpdir, f"checkpoint_{step}.pt")
                ddp_model.module.image_encoder.save(model_path)

            if (
                step % print_every == 0
                and ((i + 1) % args.num_grad_accumulation == 0)
                and is_main_process()
            ):
                percent_complete = 100 * i / len(ddp_loader)

                _, preds = torch.max(logits, 1)
                correct = torch.sum(preds == labels).item()
                accuracy = correct / labels.size(0)

                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"  # noqa: E501
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",  # noqa: E501
                    f"Acc: {accuracy}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",  # noqa: E501
                    flush=True,
                )
                run.log(
                    {
                        "step": step,
                        "train_loss": loss.item(),
                        "train_accuracy": accuracy,
                    }
                )

    # FIXME: Make this work with DDP.
    if is_main_process():
        # We only need to evaluate the model on the first GPU.
        image_encoder = ddp_model.module.image_encoder
        eval_single_dataset(image_encoder, train_dataset, args)

    if args.save is not None and is_main_process():
        zs_path = os.path.join(ckpdir, "zeroshot.pt")
        ft_path = os.path.join(ckpdir, "finetuned.pt")
        image_encoder.save(ft_path)
        return zs_path, ft_path

    cleanup_ddp()


if __name__ == "__main__":
    args = parse_arguments()
    dataset = args.train_dataset

    args.train_dataset = dataset + "Val"

    # We use gradient accumulation to simulate larger batch sizes if the model does not fit in memory.
    args.batch_size = 16 if args.model == "ViT-L-14" else 128
    args.num_grad_accumulation = 8 if args.model == "ViT-L-14" else 1

    # ─────────────────────────────────────────────────────
    # fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.lr = 1e-5
    args.wd = 0.1
    args.ls = 0.0
    args.use_timm_aug = False

    assert args.save is not None, "Please provide a save directory."
    args.save = os.path.join(args.save, f"checkpoints_{args.seed}", args.model)

    print("=" * 100)
    print(f"Finetuning {args.model} on {dataset}")
    print("=" * 100)

    group = "{}_{}".format(args.seed, time.strftime("%Y%m%d-%H%M%S"))

    # Set world_size based on the number of GPUs
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if args.world_size > n_gpus:
            print(
                f"WARNING: world_size ({args.world_size}) is greater than the number "
                f"of available GPUs ({n_gpus}). Setting world_size to {n_gpus}."
            )
            args.world_size = n_gpus
    else:
        args.world_size = 1

    torch.multiprocessing.spawn(
        finetune, args=(args, group), nprocs=args.world_size
    )
