## Guide for using Optuna

In order to run an optuna study you need to fill a config file `.json` in the `./optuna` directory.
The config file should be similar to the `template_study.json`.a

```
{
  "study_name": "study_name",
  "model_lr": "OPTUNA",
  "gnn_lr": "OPTUNA",
  "epochs": 3,
  "layer_with_gnn_1": "OPTUNA",
  "layer_with_gnn_2": "OPTUNA",
  "layer_with_gnn_3": "OPTUNA",
  "gnn_topk":"OPTUNA",
  "checkpoint_sentence_transformer": "OPTUNA",
  "reprojection_activation": "OPTUNA",
  "mysql_server_url": "mysql://optuna@IPHOST/",
  "number_of_trials": 10,
  "number_of_parallel_jobs": 4
}
```

### In depth

* `study_name`: name of the study
* `model_lr`: learning rate for the summarizer model
* `gnn_lr`: learning rate for the gnn model
* `epochs`: max number of epochs
* `layer_with_gnn_1`: the number of the first layer with gnn integration
* `layer_with_gnn_2`: the number of the second layer with gnn integration
* `layer_with_gnn_3`: the number of the third layer with gnn integration
* `gnn_topk`: number of topk for the gnn (reasoning path for each root words)
* `checkpoint_sentence_transformer`: checkpoint for the sentence transformer to encode the node and edges of the graph
* `reprojection_activation`: activation function for the reprojection layers in the gnn
* `mysql_server_url`: url of the mysql server
* `number_of_trials`: number of trials for any job in the study
* `number_of_parallel_jobs`: number of parallel jobs for the study (total number of trials will be number_of_trials * number_of_jobs trials)

The value `OPTUNA` means the optuna will decide and tune that parameters.

If you want use less gnn layer than three you must disable the layer using -1

All the parameters not specified in the config file will be used with default values.


### Run the study

To run the study you need to run the following command:

```
python3 optuna_integration.py --config_file ./optuna/study_name.json
```


### The mysql server

First you need to install the mysql server
```
apt-get install mysql-server
```

Then you need to create a user and a database for optuna that can reach the server from any host, without password
```
mysql -u root -p
> CREATE USER 'optuna'@'*' IDENTIFIED BY '';
```

Then you have to create the database for optuna and grant to the user all the privileges on the database
```
mysql -u root -e "CREATE DATABASE IF NOT EXISTS optuna_example"
mysql -u root -p
> GRANT ALL PRIVILEGES ON optuna_example.* TO 'optuna'@'*';
```

Finally you should disable the default behaviour to listen only on local host.
You have to go to `/etc/mysql/mysql.conf.d/mysqld.cnf` and comment the line `bind-address = 127.0.0.1`

You can restart the mysql server
```
sudo service mysql restart
```

Now, the server side is all set, you can go on the docker container and run:
```
mysql -u optuna -h 137.204.107.153
```

You should be able to log into the mysql server running on the host.

### Run the study within docker

Once the mysql server is up, you should create an optuna distributed study. You can do it running this command:
```
optuna create-study --study-name "distributed-example" --storage "mysql://optuna@137.204.107.153/optuna_example"
```

You should get something like this:
```
root@9dd864a506f6:/workspace# optuna create-study --study-name "distributed-example" --storage "mysql://optuna@137.204.107.153/optuna_example"
[I 2023-06-20 10:42:57,011] A new study created in RDB with name: distributed-example
distributed-example
```

You can check the database:
```
mysql -u optuna -h HOSTIP
mysql> use optuna_example;
mysql> show tables;
```

You should see several tables created by optuna.
