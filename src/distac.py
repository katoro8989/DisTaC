import os
import time

import torch
import torch.nn.functional as F
import wandb
import numpy as np
import copy
from args import parse_arguments
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from distributed import (
    cleanup_ddp,
    distribute_loader,
    is_main_process,
    setup_ddp,
)
from eval import calculate_entropy_manual
from heads import get_classification_head
from modeling import ImageClassifier, ImageEncoder
from utils import cosine_lr
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
    ckpdir = os.path.join(args.save, args.eval_dataset + "Val")

    ft_path = os.path.join(args.save, args.eval_dataset + "Val", "finetuned_distac.pt")

    tchr_path = os.path.join(args.save, args.eval_dataset + "Val", "finetuned.pt")

    tchr_state_dict = torch.load(tchr_path)
    zs_path = os.path.join(args.save, args.eval_dataset + "Val", "zeroshot.pt")

    zs_state_dict = torch.load(zs_path)

    assert train_dataset is not None, "Please provide a training dataset."

    print("Building image encoder.")
    aug_cfg = {
        "use_timm": args.use_timm_aug,
    }
    tchr_image_encoder = ImageEncoder.load(args.model, tchr_path, aug_cfg=aug_cfg)

    st_image_encoder = copy.deepcopy(tchr_image_encoder)
    st_initial_state_dict = tchr_image_encoder.state_dict().copy()
    for name, param in tchr_state_dict.items():
        st_initial_state_dict["model." + name] = zs_state_dict[name] + args.norm_lambda * (param - zs_state_dict[name])

    st_image_encoder.load_state_dict(st_initial_state_dict)

    classification_head = get_classification_head(args, train_dataset)

    tchr_model = ImageClassifier(tchr_image_encoder, classification_head)
    st_model = ImageClassifier(st_image_encoder, classification_head)

    tchr_model.freeze_head()
    st_model.freeze_head()
    tchr_model = tchr_model.cuda()
    st_model = st_model.cuda()

    preprocess_fn = tchr_model.train_preprocess
    print_every = 100

    dataset = get_dataset(
        train_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
    )

    data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)
    num_batches = len(data_loader)

    ddp_loader = distribute_loader(data_loader)
    ddp_tchr_model = torch.nn.parallel.DistributedDataParallel(
        tchr_model,
        device_ids=[rank],
        find_unused_parameters=True,
        output_device=rank,
    )
    ddp_tchr_model.eval()
    for p in ddp_tchr_model.parameters():
        p.requires_grad = False

    ddp_st_model = torch.nn.parallel.DistributedDataParallel(
        st_model,
        device_ids=[rank],
        find_unused_parameters=True,
        output_device=rank,
    )

    params = [p for p in ddp_st_model.parameters() if p.requires_grad]
    optimizer = build_optimizer(args, params)

    scheduler = cosine_lr(
        optimizer,
        args.lr,
        args.warmup_length,
        args.epochs * num_batches // args.num_grad_accumulation,
    )

    classification_head_eval = get_classification_head(args, args.eval_dataset + "Val")
    model_to_eval = ImageClassifier(st_image_encoder, classification_head_eval)
    model_to_eval.eval()
    model_to_eval.cuda()

    device = torch.device(f"cuda:{rank}")
    st_initial_state_dict = {k: v.to(device) for k, v in st_initial_state_dict.items()}

    beta = args.distil_beta

    for epoch in range(args.epochs):
        ddp_st_model.train()

        for i, batch in enumerate(ddp_loader):
            start_time = time.time()

            step = (
                i // args.num_grad_accumulation
                + epoch * num_batches // args.num_grad_accumulation
            )

            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()
            data_time = time.time() - start_time

            with torch.no_grad():
                tchr_logits = ddp_tchr_model(inputs)
            st_logits = ddp_st_model(inputs)

            soft_targets_teacher = F.softmax(tchr_logits / args.distil_T_tchr, dim=-1)
            log_softmax_student = F.log_softmax(st_logits / args.distil_T_stu, dim=-1)
            
            loss_soft = F.kl_div(log_softmax_student, soft_targets_teacher.detach(), reduction='batchmean') * (args.distil_T_stu * args.distil_T_tchr)

            loss = loss_soft
            loss.backward()

            del tchr_logits, st_logits, soft_targets_teacher, log_softmax_student

            with torch.no_grad():
                for p, p0 in zip(ddp_st_model.module.image_encoder.parameters(), st_initial_state_dict.values()):
                    if p.grad is not None:
                        p.grad.add_(2 * beta * (p - p0))

            if (i + 1) % args.num_grad_accumulation == 0:
                scheduler(step)

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            batch_time = time.time() - start_time

            if i % 50 == 0:
                torch.cuda.empty_cache()

            if (
                args.checkpoint_every > 0
                and step % args.checkpoint_every == 0
                and is_main_process()
            ):
                print("Saving checkpoint.")
                model_path = os.path.join(ckpdir, f"checkpoint_{step}.pt")
                ddp_st_model.module.image_encoder.save(model_path)

            if (
                step % print_every == 0
                and ((i + 1) % args.num_grad_accumulation == 0)
                and is_main_process()
            ):
                current_state_dict = ddp_st_model.module.image_encoder.state_dict()
                model_to_eval.image_encoder.load_state_dict(current_state_dict)
                percent_complete = 100 * i / len(ddp_loader)

                current_weights = np.concatenate([p.cpu().numpy().flatten() for p in current_state_dict.values()])
                zero_shot_weights = np.concatenate([w.cpu().numpy().flatten() for w in zs_state_dict.values()])
                delta_weights = current_weights - zero_shot_weights
                total_norm = np.linalg.norm(delta_weights)

                entropy_stu = calculate_entropy_manual(st_logits).sum().item()

                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{num_batches}]\t"  # noqa: E501
                    f"Finlal Layer Loss: {loss_soft.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",  # noqa: E501
                    f"Total norm: {total_norm}",
                    f"Prediction Entropy: {entropy_stu}",
                    flush=True,
                )
                run.log(
                    {
                        "step": step,
                        "train_loss": loss.item(),
                        "total_norm": total_norm,
                        "prediction_entropy": entropy_stu,
                    }
                )
            
            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    if args.save is not None and is_main_process():
        image_encoder = ddp_st_model.module.image_encoder
        image_encoder.save(ft_path)
        return zs_path, ft_path

    cleanup_ddp()


if __name__ == "__main__":
    args = parse_arguments()
    dataset = args.train_dataset

    args.train_dataset = dataset + "Val"

    args.batch_size = 8 if args.model == "ViT-L-14" else 128
    args.num_grad_accumulation = 16 if args.model == "ViT-L-14" else 1

    args.lr = 1e-5
    args.wd = 0.0
    args.ls = 0.0
    args.use_timm_aug = False
    args.max_steps = 500

    assert args.save is not None, "Please provide a save directory."
    args.save = os.path.join(args.save, f"checkpoints_{args.seed}", args.model)

    print("=" * 100)
    print(f"Finetuning {args.model} on {dataset}")
    print("=" * 100)

    group = "{}_{}".format(args.seed, time.strftime("%Y%m%d-%H%M%S"))

    torch.multiprocessing.spawn(
        finetune, args=(args, group), nprocs=args.world_size
    )
