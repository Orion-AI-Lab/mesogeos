from lit_module import plUNET
from datamodule import TheDataModule
import pytorch_lightning as pl
import argparse
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from wandb_callbacks import AddMetricAggs, LogValPredictionsSegmentation, LogTestPredictionsSegmentation

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# a function to read plUNET parameters as program arguments
def add_litmodule_args(parent_parser):
    parser = parent_parser.add_argument_group("plUNET")
    parser.add_argument("--lr", type=float, default=0.01, help='Learning rate for the optimizer')
    parser.add_argument("--weight_decay", type=float, default=0.00001, help='Weight decay for the optimizer')
    parser.add_argument("--loss", type=str, default='ce', help='Loss function to use, (ce, dice)')
    parser.add_argument("--encoder_name", type=str, default='efficientnet-b5', help='Encoder name, check segmentation_models_pytorch for available encoders')
    return parent_parser


def add_datamodule_args(parent_parser):
    parser = parent_parser.add_argument_group("DataModule")
    parser.add_argument("--dataset_path", type=str, help='Path to the dataset')
    parser.add_argument("--batch_size", type=int, default=64, help='Batch size')
    parser.add_argument("--wandb_entity", type=str, help='Wandb entity name')
    parser.add_argument("--experiment_name", type=str, default='test', help='Wandb experiment name')
    parser.add_argument("--project_name", type=str, default='mesogeos_ba_unet', help='Wandb project name')
    parser.add_argument("--input_vars", type=str, default='ignition_points,ndvi,roads_distance,slope,smi', help='Input variables comma separated')
    parser.add_argument("--auto_lr_find", type=str, default='False', help='Whether to run lr finder or not')
    # parser.add_argument("--include_negatives", type=bool, default=False)
    return parent_parser


if __name__ == '__main__':
    # create the top-level parser
    parser = argparse.ArgumentParser()
    # add arguments
    parser = add_litmodule_args(parser)
    parser = add_datamodule_args(parser)
    # add argument for max_epochs
    parser.add_argument("--max_epochs", type=int, default=10)
    # parse the arguments
    hparams = parser.parse_args()

    # transform input_vars to list
    input_vars = hparams.input_vars.split(',')
    print("Input variables: ", input_vars)

    # run the main function
    print(hparams)
    pl.seed_everything(42)
    # create the datamodule from the appropriate parsed arguments
    dm = TheDataModule(
            dataset_path = hparams.dataset_path,
            input_vars = input_vars,
            target = 'burned_areas',
            batch_size = hparams.batch_size,
            num_workers = 16,
            pin_memory = True,
    )
    # create the plUNET from the appropriate parsed arguments
    model = plUNET(
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
        input_vars=input_vars,
        loss=hparams.loss,
        encoder_name=hparams.encoder_name,
    )



    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val/loss", mode="min", every_n_epochs=5)

    add_metric_aggs_cb = AddMetricAggs(['val/loss', 'val/auprc'], ['min', 'max'])
    log_val_cb = LogValPredictionsSegmentation(num_samples=16, every_n_epochs=1)
    log_test_cb = LogTestPredictionsSegmentation(num_samples=32)
    callbacks = [checkpoint_callback, add_metric_aggs_cb, log_val_cb, log_test_cb]
    wandb_logger = WandbLogger(entity='deepcube', project=hparams.project_name, name=hparams.experiment_name)

    # create the trainer with the appropriate parsed arguments
    trainer = pl.Trainer(max_epochs=hparams.max_epochs, gpus=1, logger=wandb_logger, callbacks=callbacks)

    auto_lr_find = str2bool(hparams.auto_lr_find)
    if auto_lr_find:
        lr_finder = trainer.tuner.lr_find(model, dm)
        print(lr_finder.results)
        # Plot with
        fig = lr_finder.plot(suggest=True)
        # save fig to wandb
        wandb_logger.experiment.log({'lr_finder': fig})
        # log best lr
        wandb_logger.experiment.log({'best_lr': lr_finder.suggestion()})

    wandb_logger.experiment.config.update(hparams)
    # train_epoch the model
    trainer.fit(model, dm)

    # test the model
    trainer.test(model, datamodule=dm, ckpt_path='best')