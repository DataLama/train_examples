
def get_unique_identifier(length: int = 8) -> str:
    """Create a unique identifier by choosing `length`
    random characters from list of ascii characters and numbers
    """
    alphabet = string.ascii_lowercase + string.digits
    uuid = "".join(alphabet[ix]
                   for ix in np.random.choice(len(alphabet), length))
    return uuid


def create_experiment_dir(checkpoint_dir: pathlib.Path,
                          all_arguments: Dict[str, Any]) -> pathlib.Path:
    """Create an experiment directory and save all arguments in it.
    Additionally, also store the githash and gitdiff. Finally create
    a directory for `Tensorboard` logs. The structure would look something
    like
        checkpoint_dir
            `-experiment-name
                |- hparams.json
                |- githash.log
                |- gitdiff.log
                `- tb_dir/

    Args:
        checkpoint_dir (pathlib.Path):
            The base checkpoint directory
        all_arguments (Dict[str, Any]):
            The arguments to save

    Returns:
        pathlib.Path: The experiment directory
    """
    # experiment name follows the following convention
    # {exp_type}.{YYYY}.{MM}.{DD}.{HH}.{MM}.{SS}.{uuid}
    current_time = datetime.datetime.now(pytz.timezone("US/Pacific"))
    expname = "bert_pretrain.{0}.{1}.{2}.{3}.{4}.{5}.{6}".format(
        current_time.year,
        current_time.month,
        current_time.day,
        current_time.hour,
        current_time.minute,
        current_time.second,
        get_unique_identifier(),
    )
    exp_dir = checkpoint_dir / expname
    if not is_rank_0():
        return exp_dir
    exp_dir.mkdir(exist_ok=False)
    hparams_file = exp_dir / "hparams.json"
    with hparams_file.open("w") as handle:
        json.dump(obj=all_arguments, fp=handle, indent=2)
    # Save the git hash
    #try:
    #    gitlog = sh.git.log("-1", format="%H", _tty_out=False, _fg=False)
    #    with (exp_dir / "githash.log").open("w") as handle:
    #        handle.write(gitlog.stdout.decode("utf-8"))
    #except sh.ErrorReturnCode_128:
    #    log_dist(
    #        "Seems like the code is not running from"
    #        " within a git repo, so hash will"
    #        " not be stored. However, it"
    #        " is strongly advised to use"
    #        " version control.",
    #        ranks=[0],
    #        level=logging.INFO)
    # And the git diff
    #try:
    #    gitdiff = sh.git.diff(_fg=False, _tty_out=False)
    #    with (exp_dir / "gitdiff.log").open("w") as handle:
    #        handle.write(gitdiff.stdout.decode("utf-8"))
    #except sh.ErrorReturnCode_129:
    #    log_dist(
    #        "Seems like the code is not running from"
    #        " within a git repo, so diff will"
    #        " not be stored. However, it"
    #        " is strongly advised to use"
    #        " version control.",
    #        ranks=[0],
    #        level=logging.INFO)
    # Finally create the Tensorboard Dir
    tb_dir = exp_dir / "tb_dir"
    tb_dir.mkdir(exist_ok=False)
    return exp_dir


######################################################################
################ Checkpoint Related Functions ########################
######################################################################


def load_model_checkpoint(
    load_checkpoint_dir: pathlib.Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Tuple[int, torch.nn.Module, torch.optim.Optimizer]:
    """Loads the optimizer state dict and model state dict from the load_checkpoint_dir
    into the passed model and optimizer. Searches for the most recent checkpoint to
    load from

    Args:
        load_checkpoint_dir (pathlib.Path):
            The base checkpoint directory to load from
        model (torch.nn.Module):
            The model to load the checkpoint weights into
        optimizer (torch.optim.Optimizer):
            The optimizer to load the checkpoint weigths into

    Returns:
        Tuple[int, torch.nn.Module, torch.optim.Optimizer]:
            The checkpoint step, model with state_dict loaded and
            optimizer with state_dict loaded

    """
    log_dist(
        f"Loading model and optimizer checkpoint from {load_checkpoint_dir}",
        ranks=[0],
        level=logging.INFO)
    checkpoint_files = list(
        filter(
            lambda path: re.search(r"iter_(?P<iter_no>\d+)\.pt", path.name) is
            not None,
            load_checkpoint_dir.glob("*.pt"),
        ))
    assert len(checkpoint_files) > 0, "No checkpoints found in directory"
    checkpoint_files = sorted(
        checkpoint_files,
        key=lambda path: int(
            re.search(r"iter_(?P<iter_no>\d+)\.pt", path.name).group("iter_no")
        ),
    )
    latest_checkpoint_path = checkpoint_files[-1]
    checkpoint_step = int(
        re.search(r"iter_(?P<iter_no>\d+)\.pt",
                  latest_checkpoint_path.name).group("iter_no"))

    state_dict = torch.load(latest_checkpoint_path)
    model.load_state_dict(state_dict["model"], strict=True)
    optimizer.load_state_dict(state_dict["optimizer"])
    log_dist(
        f"Loading model and optimizer checkpoints done. Loaded from {latest_checkpoint_path}",
        ranks=[0],
        level=logging.INFO)
    return checkpoint_step, model, optimizer


######################################################################
######################## Driver Functions ############################
######################################################################


def train(
        checkpoint_dir: str = None,
        load_checkpoint_dir: str = None,
        # Dataset Parameters
        mask_prob: float = 0.15,
        random_replace_prob: float = 0.1,
        unmask_replace_prob: float = 0.1,
        max_seq_length: int = 512,
        tokenizer: str = "roberta-base",
        # Model Parameters
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 512,
        h_dim: int = 256,
        dropout: float = 0.1,
        # Training Parameters
        batch_size: int = 8,
        num_iterations: int = 10000,
        checkpoint_every: int = 1000,
        log_every: int = 10,
        local_rank: int = -1,
        dtype: str = "bf16",
) -> pathlib.Path:
    """Trains a [Bert style](https://arxiv.org/pdf/1810.04805.pdf)
    (transformer encoder only) model for MLM Task

    Args:
        checkpoint_dir (str):
            The base experiment directory to save experiments to
        mask_prob (float, optional):
            The fraction of tokens to mask. Defaults to 0.15.
        random_replace_prob (float, optional):
            The fraction of masked tokens to replace with random token.
            Defaults to 0.1.
        unmask_replace_prob (float, optional):
            The fraction of masked tokens to leave unchanged.
            Defaults to 0.1.
        max_seq_length (int, optional):
            The maximum sequence length of the examples. Defaults to 512.
        tokenizer (str, optional):
            The tokenizer to use. Defaults to "roberta-base".
        num_layers (int, optional):
            The number of layers in the Bert model. Defaults to 6.
        num_heads (int, optional):
            Number of attention heads to use. Defaults to 8.
        ff_dim (int, optional):
            Size of the intermediate dimension in the FF layer.
            Defaults to 512.
        h_dim (int, optional):
            Size of intermediate representations.
            Defaults to 256.
        dropout (float, optional):
            Amout of Dropout to use. Defaults to 0.1.
        batch_size (int, optional):
            The minibatch size. Defaults to 8.
        num_iterations (int, optional):
            Total number of iterations to run the model for.
            Defaults to 10000.
        checkpoint_every (int, optional):
            Save checkpoint after these many steps.

            ..note ::

                You want this to be frequent enough that you can
                resume training in case it crashes, but not so much
                that you fill up your entire storage !

            Defaults to 1000.
        log_every (int, optional):
            Print logs after these many steps. Defaults to 10.
        local_rank (int, optional):
            Which GPU to run on (-1 for CPU). Defaults to -1.

    Returns:
        pathlib.Path: The final experiment directory

    """
    device = (torch.device(get_accelerator().device_name(), local_rank) if (local_rank > -1)
              and get_accelerator().is_available() else torch.device("cpu"))
    ################################
    ###### Create Exp. Dir #########
    ################################
    if checkpoint_dir is None and load_checkpoint_dir is None:
        log_dist(
            "Need to specify one of checkpoint_dir"
            " or load_checkpoint_dir",
            ranks=[0],
            level=logging.ERROR)
        return
    if checkpoint_dir is not None and load_checkpoint_dir is not None:
        log_dist(
            "Cannot specify both checkpoint_dir"
            " and load_checkpoint_dir",
            ranks=[0],
            level=logging.ERROR)
        return
    if checkpoint_dir:
        log_dist("Creating Experiment Directory",
                 ranks=[0],
                 level=logging.INFO)
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        all_arguments = {
            # Dataset Params
            "mask_prob": mask_prob,
            "random_replace_prob": random_replace_prob,
            "unmask_replace_prob": unmask_replace_prob,
            "max_seq_length": max_seq_length,
            "tokenizer": tokenizer,
            # Model Params
            "num_layers": num_layers,
            "num_heads": num_heads,
            "ff_dim": ff_dim,
            "h_dim": h_dim,
            "dropout": dropout,
            # Training Params
            "batch_size": batch_size,
            "num_iterations": num_iterations,
            "checkpoint_every": checkpoint_every,
        }
        exp_dir = create_experiment_dir(checkpoint_dir, all_arguments)
        log_dist(f"Experiment Directory created at {exp_dir}",
                 ranks=[0],
                 level=logging.INFO)
    else:
        log_dist("Loading from Experiment Directory",
                 ranks=[0],
                 level=logging.INFO)
        load_checkpoint_dir = pathlib.Path(load_checkpoint_dir)
        assert load_checkpoint_dir.exists()
        with (load_checkpoint_dir / "hparams.json").open("r") as handle:
            hparams = json.load(handle)
        # Set the hparams
        # Dataset Params
        mask_prob = hparams.get("mask_prob", mask_prob)
        tokenizer = hparams.get("tokenizer", tokenizer)
        random_replace_prob = hparams.get("random_replace_prob",
                                          random_replace_prob)
        unmask_replace_prob = hparams.get("unmask_replace_prob",
                                          unmask_replace_prob)
        max_seq_length = hparams.get("max_seq_length", max_seq_length)
        # Model Params
        ff_dim = hparams.get("ff_dim", ff_dim)
        h_dim = hparams.get("h_dim", h_dim)
        dropout = hparams.get("dropout", dropout)
        num_layers = hparams.get("num_layers", num_layers)
        num_heads = hparams.get("num_heads", num_heads)
        # Training Params
        batch_size = hparams.get("batch_size", batch_size)
        _num_iterations = hparams.get("num_iterations", num_iterations)
        num_iterations = max(num_iterations, _num_iterations)
        checkpoint_every = hparams.get("checkpoint_every", checkpoint_every)
        exp_dir = load_checkpoint_dir
    # Tensorboard writer
    if is_rank_0():
        tb_dir = exp_dir / "tb_dir"
        assert tb_dir.exists()
        summary_writer = SummaryWriter(log_dir=tb_dir)
    ################################
    ###### Create Datasets #########
    ################################
    log_dist("Creating Datasets", ranks=[0], level=logging.INFO)
    data_iterator = create_data_iterator(
        mask_prob=mask_prob,
        random_replace_prob=random_replace_prob,
        unmask_replace_prob=unmask_replace_prob,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
    )
    log_dist("Dataset Creation Done", ranks=[0], level=logging.INFO)
    ################################
    ###### Create Model ############
    ################################
    log_dist("Creating Model", ranks=[0], level=logging.INFO)
    model = create_model(
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        h_dim=h_dim,
        dropout=dropout,
    )
    log_dist("Model Creation Done", ranks=[0], level=logging.INFO)
    ################################
    ###### DeepSpeed engine ########
    ################################
    log_dist("Creating DeepSpeed engine", ranks=[0], level=logging.INFO)
    assert (dtype == 'fp16' or dtype == 'bf16')
    ds_config = {
        "train_micro_batch_size_per_gpu": batch_size,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-4
            }
        },
        dtype: {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 1,
            "offload_optimizer": {
                "device": "cpu"
            }
        }
    }
    model, _, _, _ = deepspeed.initialize(model=model,
                                          model_parameters=model.parameters(),
                                          config=ds_config)
    log_dist("DeepSpeed engine created", ranks=[0], level=logging.INFO)
    ################################
    #### Load Model checkpoint #####
    ################################
    start_step = 1
    if load_checkpoint_dir is not None:
        _, client_state = model.load_checkpoint(load_dir=load_checkpoint_dir)
        checkpoint_step = client_state['checkpoint_step']
        start_step = checkpoint_step + 1

    ################################
    ####### The Training Loop ######
    ################################
    log_dist(
        f"Total number of model parameters: {sum([p.numel() for p in model.parameters()]):,d}",
        ranks=[0],
        level=logging.INFO)
    model.train()
    losses = []
    for step, batch in enumerate(data_iterator, start=start_step):
        if step >= num_iterations:
            break
        # Move the tensors to device
        for key, value in batch.items():
            batch[key] = value.to(device)
        # Forward pass
        loss = model(**batch)
        # Backward pass
        model.backward(loss)
        # Optimizer Step
        model.step()
        losses.append(loss.item())
        if step % log_every == 0:
            log_dist("Loss: {0:.4f}".format(np.mean(losses)),
                     ranks=[0],
                     level=logging.INFO)
            if is_rank_0():
                summary_writer.add_scalar(f"Train/loss", np.mean(losses), step)
        if step % checkpoint_every == 0:
            model.save_checkpoint(save_dir=exp_dir,
                                  client_state={'checkpoint_step': step})
            log_dist("Saved model to {0}".format(exp_dir),
                     ranks=[0],
                     level=logging.INFO)
    # Save the last checkpoint if not saved yet
    if step % checkpoint_every != 0:
        model.save_checkpoint(save_dir=exp_dir,
                              client_state={'checkpoint_step': step})
        log_dist("Saved model to {0}".format(exp_dir),
                 ranks=[0],
                 level=logging.INFO)

    return exp_dir


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(0)
    random.seed(0)
    fire.Fire(train)
