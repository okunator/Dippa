from ray import tune


class TuneReportCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        tune.report(
            loss=trainer.callback_metrics["avg_val_loss"].item(),
            mean_accuracy=trainer.callback_metrics["avg_val_accuracy"].item())


class CheckpointCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        with tune.checkpoint_dir(step=trainer.global_step) as checkpoint_dir:
            trainer.save_checkpoint(os.path.join(checkpoint_dir, "checkpoint"))            
            
            
def train_tune_checkpoint(
    training_args,
    dataset_args,
    experiment_args,
    checkpoint_dir=None,
    num_epochs=10,
    num_gpus=0):
    
    tt_logger = TestTubeLogger(
        save_dir=tune.get_trial_dir(),
        name=config.experiment_args.model_name,
        version=config.experiment_args.experiment_version
    )
    
    trainer = Trainer(
        default_root_dir=config.experiment_args.experiment_root_dir,
        max_epochs=config.training_args.num_epochs,
        gpus=config.training_args.num_gpus,  
        logger=tt_logger,
        progress_bar_refresh_rate=0,
        callbacks=[CheckpointCallback(), TuneReportCallback()],
        profiler=True
    )
    
    # Get the model from checkpoint or from 0
    if checkpoint_dir:
        base_model = smp.Unet(
            encoder_name="resnext50_32x4d", 
            classes=2
        )
        
        pl_model = SegModel(
            base_model, 
            config.dataset_args,
            config.experiment_args,
            config.training_args
        )
        
        # get the ckpt
        checkpoint = pl_load(checkpoint_dir, map_location=lambda storage, loc: storage)
        #checkpoint = torch.load(checkpoint_dir, map_location = lambda storage, loc : storage)
        pl_model.load_state_dict(checkpoint['state_dict'])
        trainer.current_epoch = checkpoint["epoch"]
    else:
        base_model = smp.Unet(
            encoder_name="resnext50_32x4d", 
            classes=2
        )
        
        pl_model = SegModel(
            base_model, 
            config.dataset_args,
            config.experiment_args,
            config.training_args
        )


    trainer.fit(pl_model)
    
    
def tune_pbt(
    config, 
    num_samples=10, 
    num_epochs=10, 
    gpus_per_trial=1) -> None:
    
    train_config = {
        "edge_weight":1,
        "lr": 1e-3,
        "batch_size": 6,
    }

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="loss",
        mode="min",
        perturbation_interval=4,
        hyperparam_mutations={
            "lr": lambda: tune.loguniform(1e-4, 1e-1).func(None),
            "batch_size": [4, 8, 16],
            "edge_weight":[1.1, 1.2, 1.5, 2]
        })

    reporter = CLIReporter(
        parameter_columns=["edge_weight", "lr", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"]
    )

    tune.run(
        partial(
            train_tune_checkpoint,
            dataset_args=config.dataset_args,
            experiment_args=config.experiment_args,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial
        ),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=train_config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_pbt"
    )
