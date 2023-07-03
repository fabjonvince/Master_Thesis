import pdb
import sys
from functools import partial

from main import main, get_args
from optuna.integration import PyTorchLightningPruningCallback
import optuna
from t5 import available_reporjection_activations
import subprocess

available_sentence_tranformers_checkpoints = [
    ('all-MiniLM-L12-v2', 384),
    ('bert-base-nli-mean-tokens', 768),
    ('bert-large-nli-mean-tokens', 1024),
    ('roberta-base-nli-mean-tokens', 768),
    ('roberta-large-nli-mean-tokens', 1024),
    ('distilbert-base-nli-mean-tokens', 768),
    ('distilbert-multilingual-nli-stsb-quora-ranking', 768),
    ('distilbert-base-nli-stsb-mean-tokens', 768),
    ('all-mpnet-base-v2', 768),
    ('all-distilroberta-v1', 768),
    ('intfloat/e5-small-v2', 384),
    ('intfloat/e5-base-v2', 768),
]

# a entry is
# key : [arg_name, arg_type, arg_help, arg_range_min, arg_range_max, arg_range_step, arg_log_scale, arg_default_value]
# the key are: model_lr, epochs, gnn_lr, layer_with_gnn_1, layer_with_gnn_2, layer_with_gnn_3, skip_test, skip_train, gnn_topk, optuna_pruner_callback


HYPERPARAMS = {
    "model_lr": ['--model_lr', 'float', "the learning rate of the language model", 0.0000001, 0.01, None, True,
                 0.00005],
    "epochs": ['--epochs', 'int', "the number of epochs of the language model", 1, 5, 1, False, 3],
    "gnn_lr": ['--gnn_lr', 'float', "the learning rate of the gnn model", 0.0000001, 0.01, None, True, 0.00005],
    "layer_with_gnn_1": ['--layer_with_gnn_1', 'int', "the layer of the language model to use for the gnn", 0, 12, 1,
                         False, 3],
    "layer_with_gnn_2": ['--layer_with_gnn_2', 'int', "the layer of the language model to use for the gnn", 0, 12, 1,
                         False, 5],
    "layer_with_gnn_3": ['--layer_with_gnn_3', 'int', "the layer of the language model to use for the gnn", 0, 12, 1,
                         False, 7],
    "gnn_topk": ['--gnn_topk', 'int', "the number of topk nodes to consider for each root node", 1, 3, 1, False, 2],
    "optuna_pruner_callback": ['--optuna_pruner_callback', None, "use optuna pruner callback", None, None, None, None,
                               None],
    "checkpoint_sentence_transformer": ['--checkpoint_sentence_transformer',
                                        'int', "the checkpoint of the sentence transformer to use", 0,
                                        len(available_sentence_tranformers_checkpoints), 1, False, 0],
    "reprojection_activation": ['--reprojection_activation', 'int',
                                "the activation function to use for the reprojection layer", 0,
                                len(available_reporjection_activations), 1, False, 0],
}

OPTUNA_FLAG = 'OPTUNA'
monitor_metric = 'val_loss'

class ArgsObj:
    def __init__(self, args:dict):
        for k, v in args.items():
            setattr(self, k, v)



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
    args = dict(args)
    for k, v in args.items():
        # check if v is a string and then if optuna flag is present
        if isinstance(v, str) and OPTUNA_FLAG in v:
            if '-' in v:
                _, minv, maxv = v.split('-')
                if HYPERPARAMS[k][1] == 'int':
                    args[k] = trial.suggest_int(k, int(minv), int(maxv), step=HYPERPARAMS[k][5])
                elif HYPERPARAMS[k][1] == 'float':
                    args[k] = trial.suggest_float(k, float(minv), float(maxv), log=HYPERPARAMS[k][6])
            else:
                if HYPERPARAMS[k][1] == 'bool':
                    args[k] = trial.suggest_categorical(k, [True, False])
                elif HYPERPARAMS[k][1] == 'int':
                    args[k] = trial.suggest_int(k, HYPERPARAMS[k][3], HYPERPARAMS[k][4], step=HYPERPARAMS[k][5])
                elif HYPERPARAMS[k][1] == 'float':
                    args[k] = trial.suggest_float(k, HYPERPARAMS[k][3], HYPERPARAMS[k][4], log=HYPERPARAMS[k][6])
    # now I condense the ayer_with_gnn_1, layer_with_gnn_2, layer_with_gnn_3 into a single list named layer_with_gnn
    args['layer_with_gnn'] = sorted([args['layer_with_gnn_1'], args['layer_with_gnn_2'], args['layer_with_gnn_3']])
    # I remove the -1 from the list
    #pdb.set_trace()
    args['layer_with_gnn'] = [l for l in args['layer_with_gnn'] if l != -1]

    del args['layer_with_gnn_1']
    del args['layer_with_gnn_2']
    del args['layer_with_gnn_3']

    # fixing the sentence transformer checkpoint
    sent_tranf = available_sentence_tranformers_checkpoints[args['checkpoint_sentence_transformer']]
    args['checkpoint_sentence_transformer'] = sent_tranf[0]
    args['sentence_transformer_embedding_size'] = sent_tranf[1]

    # fixing the reprojection activation
    args['reprojection_activation'] = available_reporjection_activations[args['reprojection_activation']]

    pruner_callback = PyTorchLightningPruningCallback(trial, monitor=monitor_metric)
    args['optuna_pruner_callback'] = pruner_callback

    default_args = get_args(default=True)

    for k, v in default_args.items():
        if k not in args:
            args[k] = v
    args['no_wandb'] = True
    args['dont_save'] = True
    args['skip_test'] = True




    obj_args = ArgsObj(args)

    result = main(obj_args)
    return result


# load config from the first argument
config = load_config_from(sys.argv[1])

study_name = config['study_name']
number_of_trials = config['number_of_trials']
sampler = optuna.samplers.TPESampler()
pruner = optuna.pruners.HyperbandPruner()

study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner, study_name=study_name,
                            storage=config['mysql_server_url'], load_if_exists=True)
del config['study_name']
del config['number_of_trials']
del config['mysql_server_url']
del config['number_of_parallel_jobs']

objfunc = partial(objective, args=config.items())

# I want the optimize catch gpu memory overflow exception
study.optimize(objfunc, n_trials=number_of_trials, catch=(RuntimeError,),
               n_jobs=1)  # n_jobs=1 is needed to avoid gpu memory overflow
