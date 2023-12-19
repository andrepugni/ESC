This is the folder for Deep Neural Network Benchmarks for Selective Classification

We provide the files for running the experiments.
First we suggest to install the required packages in requirements.txt by using conda.
```
$ conda create --name esc --file requirements.txt
```
Then activate the environment using
```
$ conda activate esc
```
In the folder data, we provide a link for downloading preprocessed data used in our benchmark.
Download the data and put them in the `data/clean/DATASET_NAME` folder.

To run the parameter tuning on a specific dataset and a specific baseline run:
```
$ python tuning.py --data DATASET_NAME --meta META --trials 20
```

As this procedure is computationally expensive, 
we provide the best parameters we found running that procedure on all the datasets
in the folder `best_params`. 
To get results for the final experiments on a dataset (questions Q1-Q2-Q3) we run:
```
$ python testing.py --data DATASET_NAME --base all
```
To get the results for Q4
```
$ python testing_max_coverage.py --data DATASET_NAME --base all
```
To get the results for Q5
```
$ python testing_ood.py --data DATASET_NAME --base all
```
