# Custom GNN QA

## Description


## Installation


## Usage

### Dataset creation

`python main.py --train_samples 2 --val_samples 2 --test_samples 2 --skip_train --skip_test `

The above code creates a dataset with 2 train, test, and val samples. The dataset is stored in the `dataset` folder.
The name of the dataset is generated using this template: `f'dataset/eli5_{args.train_samples}_{args.val_samples}_{args.test_samples}_conceptnet'`

### Training



### Testing


## Credits

* Lorenzo Valgimigli, PHD student DISI, UNIBO (lorenzo.valgimigli@unibo.it)
* Fabian Vincenzi, Master Stundend UNIBO (MAIL)
* Luca Ragazzi, PHD student DISI, UNIBO (l.ragazzi@unibo.it)
* Prof. Gianluca Moro, Professor UNIBO (gianluca.moro@unibo.it)