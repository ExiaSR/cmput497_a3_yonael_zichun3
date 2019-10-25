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
import operator

import click
from pos_tagging.tagger import *
from pos_tagging.utils import *
from collections import Counter


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

def save_output(filename, tagged_sentences, dir="output/"):
    with open(os.path.join(dir, filename), "w") as output_file:
        output_data = ["{}\n\n".format(" ".join(["_\n".join(token) for token in sentence])) for sentence in tagged_sentences]
        output_file.writelines(output_data)

def deserialize_data(data_file):
    sentences_raw = [sentence_raw.split("\n") for sentence_raw in data_file.read().split("\n\n")]
    sequences = []
    for sentence in sentences_raw:
        tmp = []
        for token in sentence:
            splited_token = token.split(" ")
            if len(splited_token) == 2:
                tmp.append((splited_token[0], splited_token[1]))
        if len(tmp) > 0:
            sequences.append(tmp)
    return sequences


def analyze(tagger, test_deserialized, train_deserialized):
    test_analysis = analyze_test(test_deserialized)
    occurences_dict = test_analysis[0]
    original_sentences = test_analysis[1]

    # use our trained tagger to tage the sentences
    tagged_sentences = tagger.tag_sents(
        [[token[0] for token in sentence] for sentence in test_deserialized]
    )

    # oov analysis
    oov_report = oov_analysis(train_deserialized, test_deserialized, tagged_sentences)
    print(
        "Results on {total_sentences} sentences and {total_words} words, of which {oov_count} were unknown.".format(
            **oov_report
        )
    )
    print(
        "Unknown words right: {oov_right_count} ({0:.4f}%); wrong: {oov_wrong_count} ({1:.4f}%).".format(
            oov_report["oov_right_count"] / oov_report["oov_count"] * 100.0,
            oov_report["oov_wrong_count"] / oov_report["oov_count"] * 100.0,
            **oov_report,
        )
    )

    # collect mistagged metrics
    mistagged_data = analyze_mistagged(tagged_sentences, test_deserialized)
    mislabelled_dict = mistagged_data[0]
    tag_occurences_dict = analyze_test(mistagged_data[1])[0]

    print("\n==========================OOV Sampling========================")
    print("Sampling: randomly selected 10 sentences that contain OOV words.")
    print("Format: [(word, (gold_tag, test_tag)), (word, (gold_tag, test_tag))]")
    print("\n".join(["{}: {}".format(idx+1, str(each)) for idx, each in enumerate(oov_report["random_sample"])]))
    print("========================End of OOV Sampling======================\n")

    # confusion matrix
    gold = tag_list(test_deserialized)
    test = tag_list(tagged_sentences)
    cm = nltk.ConfusionMatrix(gold, test)
    print("\n==========================Confusion Matrix========================")
    print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))
    print("========================End of Confusion Matrix======================\n")

    # precesion and recall
    stats_report, stats_table = precesion_and_recall(set(gold + test), cm)
    print(stats_table)

    print(
        "Occurences in Predicted (tagged result)= {}".format(
            sorted(tag_occurences_dict.items(), key=operator.itemgetter(1), reverse=True)
        )
    )
    print(
        "Occurences in Test (Gold standard) = {}".format(
            sorted(occurences_dict.items(), key=operator.itemgetter(1), reverse=True)
        )
    )
    print(
        "Dictionary of mislabelled tags = {}\n".format(
            sorted(mislabelled_dict.items(), key=operator.itemgetter(1), reverse=True)
        )
    )

    return tagged_sentences


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

    train_file_name = os.path.splitext(ntpath.basename(train_file.name))[0]
    test_file_name = os.path.splitext(ntpath.basename(test_file.name))[0]
    model_name = model or "{}.{}.tagger".format(train_file_name, tagger_name)

    train_deserialized = deserialize_data(train_file)
    test_deserialized = deserialize_data(test_file)

    # train and test tagger
    tagger = Tagger.factory(tagger_name)
    tagger.train(train_deserialized)
    accuracy = tagger.test(test_deserialized)
    print(
        "Model: {}, Train file: {}, Test file: {}, Accuracy: {:.2f}%".format(
            model_name, train_file.name, test_file.name, accuracy * 100.0
        )
    )

    # save trained model to disk
    # save_object(tagger.tagger, model_name)

    # error analysis
    tagged_sentences = analyze(tagger.tagger, test_deserialized, train_deserialized)
    save_output("{}.{}-tagged.{}.txt".format(tagger_name, test_file_name, train_file_name), tagged_sentences)


if __name__ == "__main__":
    main()
