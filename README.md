# cmput497_a3_yonael_zichun3

## Prerequisites

-   Java

## How to run

```sh
# Setup python virtual environment
$ virtualenv venv --python=python3
$ source venv/bin/activate

# Install python dependencies
$ pip install -r requirements.txt

# download POS tagger
$ make bootstrap

# transform data files to use "#" as seperator
# or make format-dev to create a subet of training file
# for development
$ make format

# train models
$ make train

# test models
$ make test
```

## Authors

-   Yonael Bekele
-   Michael Lin
