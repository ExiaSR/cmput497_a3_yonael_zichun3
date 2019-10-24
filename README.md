# cmput497_a3_yonael_zichun3

## Prerequisites

-   Java
-   Python3
-   [virtualenv](https://virtualenv.pypa.io/en/latest/)

## How to run

### Stanford POS Tagger

```sh
# Setup python virtual environment
$ virtualenv venv --python=python3
$ source venv/bin/activate

# Install python dependencies
$ pip install -r requirements.txt

# transform data files to use "#" as seperator and one sentence per line
# or make format-dev to create a subet of training file
# for development
$ make format

# train models
$ make train

# test models
$ make test

# run error analysis
$ python stanford_post_analysis.py > test-stanford-output.txt
```

### HMM and Brill

```sh
# Train two HMM models on both respective testing sets and opposite testing sets
$ make test-hmm > test-hmm-output.txt

# Train two Brill models on both respective testing sets and opposite testing sets
$ make test-brill > test-brill-output.txt
```

## Authors

-   Yonael Bekele
-   Michael Lin
