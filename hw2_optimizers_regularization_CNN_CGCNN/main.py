#
import matplotlib.pyplot as plt
import argparse
import os
import torch
import torchvision # type: ignore
import torchvision.transforms as transforms # type: ignore
import hashlib
import numpy as onp
from typing import List, cast, Optional
from models import Model, MLP, CNN, CGCNN
from structures import rotate, flip
from optimizers import Optimizer, SGD, Momentum, Nesterov, Adam


class Accuracy(torch.nn.Module):
    R"""
    Accuracy module.
    """
    def forward(
        self,
        output: torch.Tensor, target: torch.Tensor,
        /,
    ) -> torch.Tensor:
        R"""
        Forward.
        """
        #
        output = torch.argmax(output, dim=1)
        return torch.sum(output == target) / len(target)


def command(
    args: argparse.Namespace, identifier: str, n: int,
    /,
    *,
    n_tabs: int,
) -> List[str]:
    R"""
    Command lines.
    """
    #
    source = "--source {:s}".format(str(args.source))
    seed = "--random-seed {:s}".format(str(args.random_seed))
    shuffle = "--shuffle-label" if args.shuffle_label else ""
    batch_size = "--batch-size {:s}".format(str(args.batch_size))
    cnn = "--cnn" if args.cnn else ""
    cgcnn = "--cgcnn" if args.cgcnn else ""
    kernel = "--kernel {:s}".format(str(args.kernel))
    stride = "--stride {:s}".format(str(args.stride))
    amprec = "--amprec" if args.amprec else ""
    optim_alg = "--optim-alg {:s}".format(str(args.optim_alg))
    lr = "--lr {:s}".format(str(args.lr))
    wd = "--l2-lambda {:s}".format(str(args.l2_lambda))
    num_epochs = "--num-epochs {:s}".format(str(args.num_epochs))
    device = "--device {:s}".format(str(args.device))
    rot_flip = "--rot-flip {:s}" if args.rot_flip else ""
    cmdargs = (
        [
            source, seed, shuffle, batch_size, cnn, cgcnn, kernel, stride,
            amprec, optim_alg, lr, wd, num_epochs, device, rot_flip,
        ]
    )

    #
    if len(args.sbatch) > 0:
        #
        (submit, name) = args.sbatch.split(":")
        submit = (
            {
                "account": "-A {:s}".format(name),
                "partition": "--partition={:s}".format(name),
            }[submit]
        )

    #
    heads = ["#!/bin/bash"]
    if len(args.sbatch) > 0:
        #
        heads.append("#SBATCH {:s}".format(submit))
    heads.append("#SBATCH --job-name={:s}".format(identifier))
    heads.append(
        "#SBATCH --output={:s}"
        .format(os.path.join("sbatch", "{:s}.stdout.txt".format(identifier))),
    )
    heads.append(
        "#SBATCH --error={:s}"
        .format(os.path.join("sbatch", "{:s}.stderr.txt".format(identifier))),
    )
    heads.append("#SBATCH --cpus-per-task=4")
    heads.append(
        "#SBATCH --gres=gpu:{:d}".format(1 if args.device == "cuda" else 0),
    )

    #
    lines = (
        [
            "/usr/bin/time -f \"Max CPU Memory: %M KB\nElapsed: %e sec\"",
            "python -u main.py",
        ]
    )
    for (i, argument) in enumerate(cmdargs):
        #
        if len(argument) == 0:
            #
            continue
        n_requires = (
            len(lines[-1]) + 1 + len(argument)
            + (2 if i < len(cmdargs) - 1 else 0)
        )
        if n_requires > n:
            #
            lines.append(" " * n_tabs + argument)
        else:
            #
            lines[-1] = "{:s} {:s}".format(lines[-1], argument)
    lines = " \\\n".join(lines).split("\n")
    return heads + lines


def evaluate(
    model: Model, criterion: torch.nn.Module,
    minibatcher: torch.utils.data.DataLoader,
    /,
    *,
    device: str,
) -> float:
    R"""
    Evaluate.
    """
    #
    model.eval()

    #
    buf_total = []
    buf_metric = []
    for (inputs, targets) in minibatcher:
        #
        inputs = inputs.to(device)
        targets = targets.to(device)

        #
        with torch.no_grad():
            #
            outputs = model.forward(inputs)
            total = len(targets)
            metric = criterion.forward(outputs, targets).item()
        buf_total.append(total)
        buf_metric.append(metric * total)
    return float(sum(buf_metric)) / float(sum(buf_total))


def train(
    model: Model, criterion: torch.nn.Module,
    minibatcher: torch.utils.data.DataLoader, optimizer: Optimizer,
    /,
    *,
    gradscaler: Optional[torch.cuda.amp.grad_scaler.GradScaler], device: str,
) -> None:
    R"""
    Train.
    """
    #
    model.train()

    #
    for (inputs, targets) in minibatcher:
        #
        inputs = inputs.to(device)
        targets = targets.to(device)

        #
        optimizer.zero_grad()
        if isinstance(optimizer, Nesterov):
            #
            optimizer.prev()
        if gradscaler is not None:
            #
            with torch.cuda.amp.autocast():
                #
                outputs = model.forward(inputs)
                loss = criterion(outputs, targets)
            gradscaler.scale(loss).backward()
            gradscaler.step(optimizer)
            gradscaler.update()
        else:
            #
            outputs = model.forward(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()


def main(*ARGS):
    R"""
    Main.
    """
    #
    parser = argparse.ArgumentParser(description="Main Execution (Homework 2)")
    parser.add_argument(
        "--sbatch",
        type=str, required=False, default="", help="Slurm queue.",
    )
    parser.add_argument(
        "--source",
        type=str, required=False, default=os.path.join("data", "mnist"),
        help="Path to the MNIST data directory.",
    )
    parser.add_argument(
        "--random-seed",
        type=int, required=False, default=47, help="Random seed.",
    )
    parser.add_argument(
        "--shuffle-label",
        action="store_true", help="Shuffle training label data.",
    )
    parser.add_argument(
        "--batch-size",
        type=int, required=False, default=-1, help="Batch size.",
    )
    parser.add_argument(
        "--cnn",
        action="store_true", help="Use CNN layers.",
    )
    parser.add_argument(
        "--cgcnn",
        action="store_true", help="Use G-Invariant CNN layers.",
    )
    parser.add_argument(
        "--kernel",
        type=int, required=False, default=5,
        help="Size of square kernel (filter).",
    )
    parser.add_argument(
        "--stride",
        type=int, required=False, default=1, help="Size of square stride.",
    )
    parser.add_argument(
        "--amprec",
        action="store_true",
        help="Use Automatically Mixed Precision instead of FP32.",
    )
    parser.add_argument(
        "--optim-alg",
        type=str, required=False,
        choices=["sgd", "momentum", "nesterov", "adam", "default"],
        default="default",
        help="Optimizer algorithm.",
    )
    parser.add_argument(
        "--lr",
        type=float, required=False, default=1e-3, help="Learning rate.",
    )
    parser.add_argument(
        "--l2-lambda",
        type=float, required=False, default=0.0,
        help="L2 regularization strength.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int, required=False, default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--device",
        type=str, required=False, default="cpu", choices=["cpu", "cuda"],
        help="Device to work on.",
    )
    parser.add_argument(
        "--rot-flip",
        action="store_true", help="Rotate and flip test randomly.",
    )
    parser.add_argument(
        "--identify",
        action="store_true", help="Get identity only.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true", help="Evaluate only.",
    )
    args = parser.parse_args() if len(ARGS) == 0 else parser.parse_args(ARGS)

    # Parse the command line arguments.
    sbatch = args.sbatch
    source = args.source
    seed = args.random_seed
    shuffle = args.shuffle_label
    batch_size = args.batch_size
    cnn = args.cnn
    cgcnn = args.cgcnn
    kernel = args.kernel
    stride = args.stride
    amprec = args.amprec
    optim_alg = args.optim_alg
    lr = args.lr
    wd = args.l2_lambda
    num_epochs = args.num_epochs
    device = args.device
    rot_flip = args.rot_flip
    identify = args.identify
    evalonly = args.eval_only

    #
    identifier = (
        hashlib.md5(
            str(
                (
                    seed, shuffle, batch_size, cnn, cgcnn, kernel, stride,
                    amprec, optim_alg, lr, wd,
                ),
            ).encode(),
        ).hexdigest()
    )
    print(
        "\x1b[103;30mDescription Hash\x1b[0m: \x1b[102;30m{:s}\x1b[0m"
        .format(identifier),
    )
    if identify:
        #
        return

    #
    if not os.path.isdir("sbatch"):
        #
        os.makedirs("sbatch", exist_ok=True)
    if not os.path.isdir("ptnnp"):
        #
        os.makedirs("ptnnp", exist_ok=True)
    if not os.path.isdir("ptlog"):
        #
        os.makedirs("ptlog", exist_ok=True)
    script = os.path.join("sbatch", "{:s}.sh".format(identifier))
    ptnnp = os.path.join("ptnnp", "{:s}.ptnnp".format(identifier))
    ptlog = os.path.join("ptlog", "{:s}.ptlog".format(identifier))

    #
    cmdlines = command(args, identifier, 79, n_tabs=4)
    print("\x1b[103;30mCommands\x1b[0m:")
    print("\n".join(cmdlines))

    #
    if len(sbatch) > 0:
        #
        print("\x1b[103;30mSlurm\x1b[0m:")
        if os.path.isfile(script):
            #
            os.remove(script)
        with open(script, "w") as file:
            #
            file.write("\n".join(cmdlines))
        os.system("sbatch {:s}".format(script))
        return
    else:
        #
        print("\x1b[103;30mTerminal\x1b[0m:")

    #
    thrng = torch.Generator("cpu")
    thrng.manual_seed(seed)
    dataset_train = (
        torchvision.datasets.MNIST(
            root=source, train=True, download=False,
            transform=transforms.ToTensor(),
        )
    )
    dataset_test = (
        torchvision.datasets.MNIST(
            root=source, train=False, download=False,
            transform=transforms.ToTensor(),
        )
    )

    #
    if shuffle:
        #
        thrng = torch.Generator("cpu")
        thrng.manual_seed(seed)
        shuffle_train = torch.randperm(len(dataset_train), generator=thrng)
        shuffle_test = torch.randperm(len(dataset_test), generator=thrng)
        dataset_train.targets = dataset_train.targets[shuffle_train]
        dataset_test.targets = dataset_test.targets[shuffle_test]
    if rot_flip:
        #
        print("Rotating and flipping randomly ...")
        thrng = torch.Generator("cpu")
        thrng.manual_seed(seed)
        ds_rot = (
            torch.randint(0, 4, (len(dataset_test),), generator=thrng).tolist()
        )
        ds_flip = (
            torch.randint(0, 4, (len(dataset_test),), generator=thrng).tolist()
        )
        for i in range(len(dataset_test)):
            #
            mat = dataset_test.data[i].numpy()
            mat = flip(rotate(mat, ds_rot[i]), ds_flip[i]).copy()
            dataset_test.data[i] = torch.from_numpy(mat)

    #
    minibatcher_train = (
        torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size if batch_size > 0 else len(dataset_train),
            shuffle=False,
        )
    )
    minibatcher_test = (
        torch.utils.data.DataLoader(
            dataset_test,
            batch_size=batch_size if batch_size > 0 else len(dataset_test),
            shuffle=False,
        )
    )

    #
    size = 28
    channels = [1, 100, 32]
    fcs = [300, 100, 10]
    kernel_size_conv = kernel
    stride_size_conv = stride
    kernel_size_pool = 2
    stride_size_pool = 2
    if cnn:
        #
        model = (
            CNN(
                size=size, channels=channels, shapes=fcs,
                kernel_size_conv=kernel_size_conv,
                stride_size_conv=stride_size_conv,
                kernel_size_pool=kernel_size_pool,
                stride_size_pool=stride_size_pool,
            )
        )
    elif cgcnn:
        #
        model = (
            CGCNN(
                size=size, channels=channels, shapes=fcs,
                kernel_size_conv=kernel_size_conv,
                stride_size_conv=stride_size_conv,
                kernel_size_pool=kernel_size_pool,
                stride_size_pool=stride_size_pool,
            )
        )
    else:
        #
        model = MLP(size=size, shapes=fcs)
    thrng = torch.Generator("cpu")
    thrng.manual_seed(seed)
    model.initialize(thrng)
    model = model.to(device)

    # paramsinitial = model.named_parameters()
    # initial_params = {}
    # for name, param in paramsinitial:
    #     initial_params[name] = param.data.clone()
    #
    gradscaler: Optional[torch.cuda.amp.grad_scaler.GradScaler]

    #
    metric = Accuracy()
    loss = torch.nn.CrossEntropyLoss()
    if optim_alg == "sgd":
        #
        optimizer = SGD(model.parameters(), lr=lr, weight_decay=wd)
    elif optim_alg == "momentum":
        #
        optimizer = Momentum(model.parameters(), lr=lr, weight_decay=wd)
    elif optim_alg == "nesterov":
        #
        optimizer = Nesterov(model.parameters(), lr=lr, weight_decay=wd)
    elif optim_alg == "adam":
        #
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        #
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    optimizer = cast(Optimizer, optimizer)
    gradscaler = torch.cuda.amp.GradScaler() if amprec else None

    #
    maxlen1 = 5
    maxlen2 = 9
    maxlen3 = 8
    print("=" * maxlen1, "=" * maxlen2, "=" * maxlen3, "=" * 4)
    print(
        "{:>{:d}s} {:>{:d}s} {:>{:d}s} {:>4s}".format(
            "Epoch", maxlen1, "Train Acc", maxlen2, "Test Acc", maxlen3,
            "Flag",
        ),
    )
    print("-" * maxlen1, "-" * maxlen2, "-" * maxlen3, "-" * 4)

    #
    ce_train = evaluate(model, loss, minibatcher_train, device=device)
    acc_train = evaluate(model, metric, minibatcher_train, device=device)
    acc_test = evaluate(model, metric, minibatcher_test, device=device)
    log = [(ce_train, acc_train, acc_test)]

    #
    if not evalonly:
        #
        acc_train_best = acc_train
        torch.save(model.state_dict(), ptnnp)
        flag = "*"
    else:
        #
        flag = ""
    print(
        "{:>{:d}s} {:>{:d}s} {:>{:d}s} {:>4s}".format(
            "0", maxlen1, "{:.6f}".format(acc_train), maxlen2,
            "{:.6f}".format(acc_test), maxlen3, flag,
        ),
    )
    if not evalonly:
        #
        torch.save(log, ptlog)

    #
    for epoch in range(1, 1 + (0 if evalonly else num_epochs)):
        #
        train(
            model, loss, minibatcher_train, optimizer,
            gradscaler=gradscaler, device=device,
        )
        ce_train = evaluate(model, loss, minibatcher_train, device=device)
        acc_train = evaluate(model, metric, minibatcher_train, device=device)
        acc_test = evaluate(model, metric, minibatcher_test, device=device)
        log.append((ce_train, acc_train, acc_test))

        #
        if acc_train > acc_train_best:
            #
            acc_train_best = acc_train
            torch.save(model.state_dict(), ptnnp)
            flag = "*"
        else:
            #
            flag = ""
        print(
            "{:>{:d}s} {:>{:d}s} {:>{:d}s} {:>4s}".format(
                str(epoch), maxlen1, "{:.6f}".format(acc_train), maxlen2,
                "{:.6f}".format(acc_test), maxlen3, flag,
            ),
        )
        torch.save(log, ptlog)
    print("=" * maxlen1, "=" * maxlen2, "=" * maxlen3, "=" * 4)

    #
    model.load_state_dict(torch.load(ptnnp))
    acc_train = evaluate(model, metric, minibatcher_train, device=device)
    acc_test = evaluate(model, metric, minibatcher_test, device=device)
    print("Train Acc: {:.6f}".format(acc_train))
    print(" Test Acc: {:.6f}".format(acc_test))
    #
    #
    # paramsfinal = model.named_parameters()
    # final_params = {}
    # for name, param in paramsfinal:
    #     final_params[name] = param.data.clone()
    #
    # v_err = []
    # alpha_v = []
    #
    # # Interpolate model parameters
    # for alpha in torch.linspace(0, 1.5, steps=10):
    #     alpha = alpha.to(device)
    #     for name, param in model.named_parameters():
    #         param.data = (1. - alpha) * initial_params[name].data + alpha * final_params[name].data
    #     v_err.append(100. - evaluate(model, metric, minibatcher_test, device=device) * 100)
    #     alpha_v.append(alpha.cpu())
    #     print(f' alpha = {alpha} has validation error {v_err[-1]}%')
    #
    # plt.xlabel("Alpha")
    # plt.ylabel("Validation Error (%)")
    # plt.semilogy(alpha_v, v_err, linestyle='-', marker='o', color='b')
    # plt.savefig("shuff_interp.png")

#
if __name__ == "__main__":
    #
    main()