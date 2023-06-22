# Custom GNN QA

## Description


## Installation

1. First you need to clone this repository:

```
git clone https://....

```


2. You need to create the docker image:

```
cd dockerfiles
docker build -t qagnn .
```



## Usage

### Dataset creation

The above code creates a dataset with 2 train, test, and val samples. The dataset is stored in the `dataset` folder.
The name of the dataset is generated using this template: `f'dataset/eli5_{args.train_samples}_{args.val_samples}_{args.test_samples}_conceptnet'`

### Training

In order to run a optuna study you need to run the following command:

```
 optuna/run_optuna_study_using_slurm.sh optuna/<config_file>.json

```

See `optuna/guide.md` for further details


### Testing


## Credits

* Lorenzo Valgimigli, PHD student DISI, UNIBO (lorenzo.valgimigli@unibo.it)
* Fabian Vincenzi, Master Stundend UNIBO (MAIL)
* Luca Ragazzi, PHD student DISI, UNIBO (l.ragazzi@unibo.it)
* Prof. Gianluca Moro, Professor UNIBO (gianluca.moro@unibo.it)