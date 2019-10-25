"""
Script to collect metrics for error analysis on Stanford POS Tagger.
Metrics are log to STDOUT

Run:
$ python3 stanford_post_analysis.py
"""


import os
import re
import operator

import nltk

from pos_tagging.utils import *
from collections import Counter


def get_files(dir="output"):
    if not os.path.isdir(dir):
        raise Exception('Directory "{}" does not exist.'.format(dir))

    (dirpath, _, filenames) = next(os.walk(dir))
    filenames = sorted([filename for filename in filenames if filename.endswith("txt")])
    files = []
    for filename in filenames:
        with open(os.path.join(dirpath, filename)) as input_f:
            data = input_f.read()
            files.append({"path": os.path.join(dirpath, filename), "name": filename, "data": data})
    return files


def deserialize_data(data):
    sentences_raw = [sentence_raw.split(" ") for sentence_raw in data.split("\n")]
    sequences = []
    for sentence in sentences_raw:
        tmp = []
        for token in sentence:
            splited_token = token.split("_")
            if len(splited_token) == 2:
                tmp.append((splited_token[0], splited_token[1]))
        if len(tmp):
            sequences.append(tmp)
    return sequences


def get_file_by_name(name, file_list):
    for each in file_list:
        if name == each["name"]:
            return each


def main():
    tagged_files = get_files("output")
    test_files = get_files("A3DataCleaned")

    for tagged_file in tagged_files:
        tmp = re.search(r"(.*)-tagged.(.*).txt", tagged_file["name"])
        test_name = tmp.group(1)
        model_name = tmp.group(2)

        tagged_sentences = deserialize_data(tagged_file["data"])
        test_sentences = deserialize_data(
            get_file_by_name("{}.cleaned.txt".format(test_name), test_files)["data"]
        )
        train_sentences = deserialize_data(
            get_file_by_name("{}.cleaned.txt".format(model_name), test_files)["data"]
        )

        print("Test dataset: {}, Model name: {}".format(test_name, model_name))
        test_analysis = analyze_test(test_sentences)
        occurences_dict = test_analysis[0]
        original_sentences = test_analysis[1]

        # oov analysis
        oov_report = oov_analysis(train_sentences, test_sentences, tagged_sentences)
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

        mistagged_data = analyze_mistagged(tagged_sentences, test_sentences)
        mislabelled_dict = mistagged_data[0]
        tag_occurences_dict = analyze_test(mistagged_data[1])[0]

        print("\n==========================OOV Sampling========================")
        print("Sampling: randomly selected 10 sentences that contain OOV words.")
        print("Format: [(word, (gold_tag, test_tag)), (word, (gold_tag, test_tag))]")
        print("\n".join(["{}: {}".format(idx+1, str(each)) for idx, each in enumerate(oov_report["random_sample"])]))
        print("========================End of OOV Sampling======================\n")

        # confusion matrix
        gold = tag_list(test_sentences)
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


if __name__ == "__main__":
    main()
