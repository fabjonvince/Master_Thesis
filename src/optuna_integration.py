import sys
from functools import partial

from src.main import main
from optuna.integration import PyTorchLightningPruningCallback
import optuna

# a entry is
# key : [arg_name, arg_type, arg_help, arg_range_min, arg_range_max, arg_range_step, arg_log_scale, arg_default_value]
# the key are: model_lr, epochs, gnn_lr, layer_with_gnn_1, layer_with_gnn_2, layer_with_gnn_3, skip_test, skip_train, gnn_topk, optuna_pruner_callback

HYPERPARAMS = {
    "model_lr": ['--model_lr', float, "the learning rate of the language model", 0.01, 0.0000001, None, True, 0.00005],
    "epochs": ['--epochs', int, "the number of epochs of the language model", 1, 5, 1, False, 3],
    "gnn_lr": ['--gnn_lr', float, "the learning rate of the gnn model", 0.01, 0.0000001, None, False, 0.00005],
    "layer_with_gnn_1": ['--layer_with_gnn_1', int, "the layer of the language model to use for the gnn", 0, 9, 1,
                         False, 3],
    "layer_with_gnn_2": ['--layer_with_gnn_2', int, "the layer of the language model to use for the gnn", 0, 9, 1,
                         False, 5],
    "layer_with_gnn_3": ['--layer_with_gnn_3', int, "the layer of the language model to use for the gnn", 0, 9, 1,
                         False, 7],
    "gnn_topk": ['--gnn_topk', int, "the number of topk nodes to consider for each root node", 1, 3, 1, False, 2],
    "skip_test": ['--skip_test', bool, "skip the test phase", None, None, None, None, True],
    "skip_train": ['--skip_train', bool, "skip the train phase", None, None, None, None, False],
    "optuna_pruner_callback": ['--optuna_pruner_callback', None, "use optuna pruner callback", None, None, None, None, None]
}

OPTUNA_FLAG = 'OPTUNA'
monitor_metric='val_loss'


def load_config_from(config_path):
    """
    Function that load a json file into a dict object.
    If some keys from HYPERPARAMS are not present in the json file, the default value is used.
    :param config_path:
    :return:
    """
    import json
    with open(config_path, "r") as f:
        config = json.load(f)
    for k, v in HYPERPARAMS.items():
        if k not in config:
            config[k] = v[-1]
    return config


def objective(trial, args):
    for k, v in args:
        if v == OPTUNA_FLAG:
            if HYPERPARAMS[k][1] == bool:
                args[k] = trial.suggest_categorical(k, [True, False])
            elif HYPERPARAMS[k][1] == int:
                args[k] = trial.suggest_int(k, HYPERPARAMS[k][3], HYPERPARAMS[k][4], step=HYPERPARAMS[k][5])
            elif HYPERPARAMS[k][1] == float:
                args[k] = trial.suggest_float(k, HYPERPARAMS[k][3], HYPERPARAMS[k][4], log=HYPERPARAMS[k][6])
    # now I condense the ayer_with_gnn_1, layer_with_gnn_2, layer_with_gnn_3 into a single list named layer_with_gnn
    args['layer_with_gnn'] = sorted([args['layer_with_gnn_1'], args['layer_with_gnn_2'], args['layer_with_gnn_3']], reverse=True)
    del args['layer_with_gnn_1']
    del args['layer_with_gnn_2']
    del args['layer_with_gnn_3']

    # Now I have to fix the optuna pruner callback
    pruner_callback = PyTorchLightningPruningCallback(trial, monitor=monitor_metric)
    args['optuna_pruner_callback'] = pruner_callback

    result = args['model_lr'] * args['gnn_lr']#main(args)
    return result

# load config from the first argument
config = load_config_from(sys.argv[1])

study_name=config['study_name']
sampler = optuna.samplers.TPESampler()
pruner = optuna.pruners.HyperbandPruner()

study = optuna.load_study(sampler=sampler, pruner=pruner, study_name=study_name, storage="mysql://optuna@localhost/" + str(study_name))
objfunc = partial(objective, args=config.items(),direction='maximize')

# I want the optimize catch gpu memory overflow exception

study.optimize(objfunc, n_trials=100, catch=(RuntimeError,), n_jobs=1) # n_jobs=1 is needed to avoid gpu memory overflow

