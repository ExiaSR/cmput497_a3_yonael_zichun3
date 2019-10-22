"""
Example: python tagger.py --train_file A3DataCleaned/Domain1Train.txt --test_file A3DataCleaned/Domain1Test.txt --tagger hmm

Usage: tagger.py [OPTIONS]

Options:
  --tagger TEXT          Tagger name, [hmm|brill]  [required]
  --model TEXT           Model file to read or write
  --train_file FILENAME  Path to train file  [required]
  --test_file FILENAME   Path to test file  [required]
  --debug                Enable debug message.
  --help                 Show this message and exit.
"""
import sys
import os
import ntpath
import dill
import logging

import click
from pos_tagging.tagger import *
from nltk import word_tokenize


logger = logging.getLogger("tagger")


def init_logger(debug):
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


def save_object(obj, filename):
    with open(filename, "wb") as output:  # Overwrites any existing file.
        dill.dump(obj, output)


def read_object(filename):
    with open(filename, "rb") as file:
        return dill.load(file)


def deserialize_data(data_file):
    sentences_raw = [sentence_raw.split("\n") for sentence_raw in data_file.read().split("\n\n")]
    sequences = []
    for sentence in sentences_raw:
        tmp = []
        for token in sentence:
            splited_token = token.split(" ")
            if len(splited_token) == 2:
                tmp.append((splited_token[0], splited_token[1]))
        sequences.append(tmp)
    return sequences


# goes through Dictionary with POS tagged sentences
# Returns :
# (1) a dictionary with occurences for each POS tag found
# (2) a list of the original sentences without their POS tags
# https://www.nltk.org/_modules/nltk/tag/hmm.html#HiddenMarkovModelTagger.test - to detag the POS tags
def analyze_test(data):
    occurences_dict = {}
    original_sentences = []
    for sentence in data:
        for word, tag in sentence:
            orig_sent = ""
            if tag not in occurences_dict:
                occurences_dict[tag] = 1

            else:
                occurences_dict[tag] += 1

        original_sentence = " ".join("%s" % token for (token, tag) in sentence)
        original_sentences.append(original_sentence)

    return occurences_dict, original_sentences


# returns a dictionary with the mistakes in the tagger, and tagged_sentences using the custom tagger
# Format = Keys are tuples of (Test, CustomerTagger), values are occurences
def analyze_mistagged(original_sentences, tagger, test_deserialized):
    tagged_sentences = []
    mislabelled = {}

    # https://stackoverflow.com/questions/43747451/stanford-nlp-tagger-via-nltk-tag-sents-splits-everything-into-chars
    # fixed bug where tagger was reading by char

    # uses our custom trained tagger on the original sentences
    tagged_sentences = tagger.tag_sents(word_tokenize(sent) for sent in original_sentences)
    count = 0
    for test, tagged in zip(tagged_sentences, test_deserialized):
        tag = str(test[1][1])
        mistagged = str(tagged[1][1])
        if tag != mistagged:
            count += 1
            if (tag, mistagged) in mislabelled:
                mislabelled[(tag, mistagged)] += 1
            else:
                mislabelled[(tag, mistagged)] = 1

    print("Number of mistagged word tokens = {}".format(count))
    return mislabelled, tagged_sentences


def analyze(tagger, test_deserialized):
    test_analysis = analyze_test(test_deserialized)
    occurences_dict = test_analysis[0]
    original_sentences = test_analysis[1]
    mistagged_data = analyze_mistagged(original_sentences, tagger, test_deserialized)
    mislabelled_dict = mistagged_data[0]
    tag_occurences_dict = analyze_test(mistagged_data[1])[0]
    print("Occurences in Custom Tagger= {}\n".format(tag_occurences_dict))
    print("Occurences in Test (Gold standard) = {}\n".format(occurences_dict))
    print("Dictionary of mislabelled tags = {}\n".format(mislabelled_dict))


@click.command()
@click.option("--tagger", "tagger_name", help="Tagger name, [hmm|brill]", required=True)
@click.option("--model", "model", help="Model file to read or write")
@click.option(
    "--train_file", "train_file", type=click.File("r"), help="Path to train file", required=True
)
@click.option(
    "--test_file", "test_file", type=click.File("r"), help="Path to test file", required=True
)
@click.option("--debug", "debug", flag_value="debug", help="Enable debug message.")
def main(tagger_name, model, train_file, test_file, debug):
    if tagger_name not in ["hmm", "brill"]:
        print('Try "tagger.py --help" for help.')
        sys.exit(1)

    init_logger(debug)

    model_name = model or "{}.{}.tagger".format(
        os.path.splitext(ntpath.basename(train_file.name))[0], tagger_name
    )

    # train and test tagger
    tagger = Tagger.factory(tagger_name)
    tagger.train(deserialize_data(train_file))
    test_deserialized = deserialize_data(test_file)
    accuracy = tagger.test(test_deserialized)
    logger.info(
        "Model: {}, Train file: {}, Test file: {}, Accuracy: {:.2f}%".format(
            model_name, train_file.name, test_file.name, accuracy * 100.0
        )
    )

    # save trained model to disk
    save_object(tagger.tagger, model_name)

    # error analysis
    analyze(tagger.tagger, test_deserialized)


if __name__ == "__main__":
    main()
