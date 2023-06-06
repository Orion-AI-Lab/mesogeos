import argparse
import torch
import datasets.dataset as module_data
import dataloaders.dataloader as module_dataloader
import models.loss as module_loss
import models.metric as module_metric
import models.model as module_arch
from parse_config import ConfigParser
from utils import MetricTracker
from logger import TensorboardWriter
from pathlib import Path
from glob import glob
import numpy as np
import torch.nn as nn
import collections



def main(config):
    logger = config.get_logger('test')

    dynamic_features = config["features"]["dynamic"]
    static_features = config["features"]["static"]

    dataset = config.init_obj('dataset', module_data,
                              dynamic_features=dynamic_features, static_features=static_features,
                              train_val_test='test')
    dataloader = config.init_obj('dataloader', module_dataloader, dataset=dataset).dataloader()

    # device, device_ids = prepare_device(config['n_gpu'], config['gpu_id'])
    device = 'cpu'
    # # build models architecture
    if config["model_type"] == "lstm":
        model = config.init_obj('arch', module_arch, input_dim=len(dynamic_features) + len(static_features),
                                output_lstm=config['model_args']['dim'], dropout=config['model_args']['dropout'])

    elif config["model_type"] == "transformer":
        model = config.init_obj('arch', module_arch, seq_len=config["dataset"]["args"]["lag"],
                                input_dim=len(dynamic_features) + len(static_features),
                                d_model=config['model_args']['model_dim'],
                                nhead=config['model_args']['nheads'],
                                dim_feedforward=config['model_args']['ff_dim'],
                                num_layers=config['model_args']['num_layers'],
                                channel_attention=False)

    elif config["model_type"] == "gtn":
        model = config.init_obj('arch', module_arch, seq_len=config["dataset"]["args"]["lag"],
                                input_dim=len(dynamic_features) + len(static_features),
                                d_model=config['model_args']['model_dim'],
                                nhead=config['model_args']['nheads'],
                                dim_feedforward=config['model_args']['ff_dim'],
                                num_layers=config['model_args']['num_layers'],
                                channel_attention=True)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metric_ftns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config["model_path"]))
    checkpoint = torch.load(config["model_path"], map_location=device)

    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare models for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    e = 0.000001
    cfg_trainer = config['trainer']
    logger = config.get_logger('trainer', config['trainer']['verbosity'])
    writer = TensorboardWriter(config.log_dir, logger, cfg_trainer['tensorboard'])
    test_metrics = MetricTracker('loss', *[m.__name__ for m in metric_ftns], writer=writer)
    test_metrics.reset()

    with torch.no_grad():
        for batch_idx, (dynamic, static, bas_size, labels) in enumerate(dataloader):
            static = static.unsqueeze(1).repeat(1, dynamic.shape[1], 1)
            labels = labels.to(device, dtype=torch.long)
            input_ = torch.cat([dynamic, static], dim=2)
            input_ = input_.to(device, dtype=torch.float32)
            bas_size = bas_size.to(device, dtype=torch.float32)
            # bas_size=1

            if config['model_type'] in ['transformer', 'gtn']:
                input_ = torch.transpose(input_, 0, 1)
            outputs = model(input_)
            m = nn.Softmax(dim=1)
            outputs = m(outputs)

            loss = criterion(torch.log(outputs + e), labels)
            loss = torch.mean(loss * bas_size)

            output = torch.argmax(outputs, dim=1)

            writer.set_step(batch_idx)
            test_metrics.update('loss', loss.item() * dynamic.size(0), dynamic.size(0))

            for met in metric_ftns:
                if met.__name__ not in ['aucpr']:
                    test_metrics.update(met.__name__, met(output, labels)[0], met(output, labels)[1])
                elif met.__name__ == 'aucpr':
                    test_metrics.aucpr_update(met.__name__, met(outputs[:, 1], labels)[0],
                                                    met(outputs[:, 1], labels)[1])

    log = test_metrics.result()
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
