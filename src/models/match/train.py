import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import wandb
from dataloader import TranslationH5Dataset
from model import GlotSentenceEmbed
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer


def compute_loss(
    model, teacher_model, inputs, labels, criterion, target_lang, return_outputs=False
):
    # forward pass
    outputs = model(inputs)
    # get sonar embeddings
    with torch.no_grad():
        encoded_labels = teacher_model.predict(
            list(labels), source_lang=f"{target_lang}_Latn"
        )
    # compute custom MSE loss
    loss = criterion(outputs, encoded_labels.clone())
    return (loss, outputs) if return_outputs else loss


def train(
    rank: int,
    world_size: int,
    language: str,
    dataset_path: str,
    freeze_model: bool = False,
    batch_size: int = 1024,
    num_epochs: int = 32,
    evaluation_steps_per_epoch=2,
    learning_rate: float = 5e-5,
    results_path: str = "temp_results",
    best_valid_loss: float = float("Inf"),
    target_lang: str = "spa",
):
    # set env vars for distributed training
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    gpu_id = rank

    results_path = os.path.join(results_path, language)
    if not os.path.exists(results_path) and gpu_id == 0:
        os.makedirs(results_path)

    # avoid problems with master gpu taking more space
    # https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
    torch.cuda.set_device(gpu_id)
    torch.cuda.empty_cache()

    # load datasets
    print(f"Loading dataset from {dataset_path}")
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("cis-lmu/glot500-base")
    train_dataset = TranslationH5Dataset(dataset_path, language, tokenizer)
    val_dataset = TranslationH5Dataset(dataset_path, language, tokenizer, is_val=True)

    # create sampler. Ensures non overlaping input batch
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=gpu_id,
        shuffle=False,
        drop_last=False,
    )

    # create dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,  # shuffling is done by the sampler
        num_workers=1,  # use only one worker to avoid multiproccess problems with DDP
        sampler=train_sampler,
    )

    val_dataloader = (
        DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=True,  # ensures shuffling after iterating over all batches
            num_workers=1,
        )
        if gpu_id == 0
        else None
    )

    # load models
    sonar_dim = 1024
    print("Loading Glot Model")
    model = GlotSentenceEmbed(
        "cis-lmu/glot500-base", freeze_model=freeze_model, sonar_embed_size=sonar_dim
    ).to(gpu_id)

    print("Loading SONAR")
    teacher_model = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=torch.device(f"cuda:{gpu_id}"),
    )
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # set distributed model
    model = DDP(
        model, device_ids=[gpu_id], output_device=gpu_id, find_unused_parameters=True
    )
    criterion = nn.MSELoss()

    # initialize counters
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    eval_every = len(train_dataloader) // evaluation_steps_per_epoch
    # initializing model
    print(
        f"""Initiating training for {num_epochs} epochs on GPU {gpu_id}...
            {len(train_dataloader)} batches in training set
            {len(val_dataloader) if gpu_id == 0 else 0} batches in validation set
            Running model on {model.device}
            Running sonar on {teacher_model.device}
          """
    )

    if gpu_id == 0:  # logging
        # initiate experiment
        wandb.init(
            project="SONAR",
            config=dict(
                language=language,
                batch_size=batch_size,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
            ),
        )
        wandb.watch(model)
        wandb.define_metric("val_loss", summary="min")

    # loop de entrenamento
    model.train()
    with tqdm(total=num_epochs * len(train_dataloader)) as pbar:
        for epoch in range(num_epochs):
            # if we are using DistributedSampler, we have to tell it which epoch this is
            # this will also do the shuffling
            train_dataloader.sampler.set_epoch(epoch)
            print(f"Running epoch {epoch}")
            for index, batch in enumerate(train_dataloader):
                inputs, labels = batch
                inputs = inputs.to(gpu_id)

                loss, predictions = compute_loss(
                    model,
                    teacher_model,
                    inputs,
                    labels,
                    criterion,
                    target_lang,
                    return_outputs=True,
                )

                optimizer.zero_grad()
                loss.backward()  # here DDP averages model param grads
                optimizer.step()

                # actualizar conteos
                running_loss += loss.item()
                global_step += 1

                # we have to create enough room to store the collected objects
                batch_losses = [None for _ in range(world_size)]
                # the first argument is the collected lists, the second argument
                # is the data unique in each process
                dist.all_gather_object(batch_losses, running_loss)

                # evaluacion
                if global_step % eval_every == 0 and gpu_id == 0:  # val only master gpu
                    with torch.no_grad():
                        # validacion
                        for inputs, labels in val_dataloader:
                            inputs = inputs.to(gpu_id)
                            loss, predictions = compute_loss(
                                model,
                                teacher_model,
                                inputs,
                                labels,
                                criterion,
                                target_lang,
                                return_outputs=True,
                            )

                            valid_running_loss += loss.item()

                    # # get average loss from each GPU
                    average_train_loss = sum(batch_losses) / (world_size * eval_every)

                    average_valid_loss = valid_running_loss / len(val_dataloader)
                    train_loss_list.append(average_train_loss)
                    valid_loss_list.append(average_valid_loss)
                    global_steps_list.append(global_step)

                    # resetear conteos de la epoca
                    running_loss = 0.0
                    valid_running_loss = 0.0
                    model.train()

                    # imprimir resultados hasta el momento
                    total_steps = num_epochs * len(train_dataloader)
                    print(
                        """ Epoch [{}/{}]
                            Step [{}/{}]
                            Train Loss: {:.4f}
                            Valid Loss: {:.4f}""".format(
                            epoch + 1,
                            num_epochs,
                            global_step,
                            total_steps,
                            average_train_loss,
                            average_valid_loss,
                        )
                    )

                    wandb.log(
                        {
                            "val_loss": average_valid_loss,
                            "train_loss": average_train_loss,
                        },
                        step=global_step,
                    )

                    # checkpoint
                    if best_valid_loss > average_valid_loss:
                        best_valid_loss = average_valid_loss
                        save_checkpoint(
                            os.path.join(results_path, "model.pt"),
                            model,
                            best_valid_loss,
                        )

                pbar.update(1)

    print("Finished Training!")
    dist.destroy_process_group()  # clean up


def save_checkpoint(save_path, model, valid_loss):
    if save_path is None:
        return
    state_dict = {"model_state_dict": model.state_dict(), "valid_loss": valid_loss}
    torch.save(state_dict, save_path)
    print(f"Model saved to ==> {save_path}")


def load_checkpoint(load_path, model, device):
    if load_path is None:
        return
    state_dict = torch.load(load_path, map_location=device)
    print(f"Model loaded from <== {load_path}")
    model.load_state_dict(state_dict["model_state_dict"])
    return state_dict["valid_loss"]


def argparser():
    parser = argparse.ArgumentParser(
        description="Script for multilingual translation models"
    )
    parser.add_argument(
        "-l",
        dest="language",
        choices=["rap", "azum", "map", "rag"],
        action="store",
        help="Language to detect",
    )

    parser.add_argument("--translation", action=argparse.BooleanOptionalAction)
    parser.add_argument("--data_folder", default="datasets", type=str)
    parser.add_argument("--results_folder", default="temp_results", type=str)
    parser.add_argument("--model_name", default="sequence_match", type=str)
    # ------------------- Trainer args -------------------
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--epochs", default=32, type=int)
    parser.add_argument("--init_lr", default=5e-5, type=float)
    parser.add_argument(
        "--freeze", action=argparse.BooleanOptionalAction, default=False
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argparser()
    dataset_path = os.path.join(args.data_folder, f"{args.language}.hdf5")
    results_path = os.path.join(args.results_folder, args.language, args.model_name)

    world_size = torch.cuda.device_count()
    print(f"Running on {world_size} GPUs")

    batch_size = int(args.batch_size / world_size)

    mp.spawn(
        train,
        args=(
            world_size,
            args.language,
            dataset_path,
            args.freeze,
            batch_size,
            args.epochs,
        ),
        nprocs=world_size,
    )
