import argparse
import os

import torch


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_location",
        type=str,
        default=os.path.expanduser("/path/to/dataset"),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--eval_datasets",
        default=None,
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma, e.g. MNIST,EuroSAT. ",
    )
    parser.add_argument(
        "--train_dataset",
        default="Cars",
        type=str,
        help="Which dataset(s) to patch on.",
    )
    parser.add_argument(
        "--eval_dataset",
        default="Cars",
        type=str,
        help="Which dataset(s) to patch on.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only.",
    )
    parser.add_argument(
        "--results_db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-32",
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--task_to_orth",
        type=str,
        default="DTD",
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=.1,
    )
    parser.add_argument(
        "--penalty_iter",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--orth_batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--num_grad_accumulation",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument("--ls", type=float, default=0.0,
                        help="Label smoothing.")
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",  # noqa: E501
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip_cachedir",
        type=str,
        default=os.path.expanduser("~/openclip-cachedir/open_clip"),
        help="Directory for caching models from OpenCLIP",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Number of processes for distributed training.",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=-1,
        help="How often to checkpoint the model.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=12355,
        help="Port for distributed training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed.",
    )
    parser.add_argument(
        "--device_number",
        type=int,
        default=0,
        help="device_number",
    )
    parser.add_argument(
        "--finetuning_mode",
        default="standard",
        choices=["standard"],
        help="Fine-tuning mode.",
    )
    parser.add_argument(
        "--n_eval_points",
        type=int,
        default=21,
        help="Number of evaluation points used to find optimal coefficient in task arithmetic.",
    )

    parser.add_argument(
        "--addition_scaling_factor",
        type=float,
        default=0.25,
        help="momentum for SGD optimizer.",
    )

    # hessian
    parser.add_argument(
        "--n_top_eigen",
        type=int,
        default=1,
    )

    # optimizer
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="Optimizer to use.",
    )
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate.")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="momentum for SGD optimizer.",
    )
    parser.add_argument(
        "--beta_1",
        type=float,
        default=0.9,
        help="beta1 for Adam optimizer.",
    )
    parser.add_argument(
        "--beta_2",
        type=float,
        default=0.999,
        help="beta2 for Adam optimizer.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="epsilon for Adam optimizer.",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.05,
        help="rho for SAM optimizer.",
    )

    parser.add_argument(
        '--use_timm_aug',
        action='store_true',
    )

    parser.add_argument(
        '--random_param',
        action='store_true',
    )
    
    # wandb

    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="test",
    )

    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="xxx",
    )

    # distac

    parser.add_argument(
        "--norm_lambda",
        type=float,
        default=0.1,
        help="Weight for the norm regularization term from zero-shot model.",
    )

    parser.add_argument(
        "--distil_T_tchr",
        type=float,
        default=1.0,
        help="Temperature for distillation.",
    )

    parser.add_argument(
        "--distil_T_stu",
        type=float,
        default=1.0,
        help="Temperature for distillation.",
    )
    
    parser.add_argument(
        "--distil_beta",
        type=float,
        default=1.0,
        help="wd for distillation.",
    )

    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]
    return parsed_args

    # test
