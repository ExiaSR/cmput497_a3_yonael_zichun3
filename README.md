# cmput497_a3_yonael_zichun3

## Prerequisites

-   Java
-   Python3
-   [virtualenv](https://virtualenv.pypa.io/en/latest/)

## How to run

### Setup

```sh
# Setup python virtual environment
$ virtualenv venv --python=python3
$ source venv/bin/activate

# Install python dependencies
$ pip install -r requirements.txt
```

### Stanford POS Tagger

```sh
# transform data files to use "_" as seperator and one sentence per line
# or make format-dev to create a subet of training file
# for development
$ make format

# train models
# NOTE to marker: model has been pre-trained and attach as a part of the submission
# you may skip this part and test directly
$ make train

# test models
# tagged sentences are saved under `output/` directory
$ make test

# run error analysis
# the output consists of accuracy, confusion metrics, percision/recall, and some other stuffs
$ python stanford_post_analysis.py > test-stanford-output.txt
```

### HMM and Brill

```sh
# Train two HMM models on both respective testing sets and opposite testing sets
# the output consists of accuracy, confusion metrics, percision/recall, and some other stuffs
$ make test-hmm > test-hmm-output.txt

# Train two Brill models on both respective testing sets and opposite testing sets
# the output consists of accuracy, confusion metrics, percision/recall, and some other stuffs
$ make test-brill > test-brill-output.txt
```

## Authors

-   Yonael Bekele
-   Michael Lin
