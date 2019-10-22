"""
Script to collect metrics for error analysis on Stanford POS Tagger.
Metrics are log to STDOUT

Run:
$ python3 stanford_post_analysis.py
"""


import os
import re

from pos_tagging.utils import *


def get_files(dir="output"):
    if not os.path.isdir(dir):
        raise Exception("Directory \"{}\" does not exist.".format(dir))

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
            splited_token = token.split("#")
            if len(splited_token) == 2:
                tmp.append((splited_token[0], splited_token[1]))
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
        test_sentences = deserialize_data(get_file_by_name("{}.cleaned.txt".format(test_name), test_files)["data"])

        test_analysis = analyze_test(test_sentences)
        occurences_dict = test_analysis[0]
        original_sentences = test_analysis[1]

        mistagged_data = analyze_mistagged(tagged_sentences, test_sentences)
        mislabelled_dict = mistagged_data[0]
        tag_occurences_dict = analyze_test(mistagged_data[1])[0]

        print("Test dataset: {}, Model name: {}".format(test_name, model_name))
        print("Occurences in Custom Tagger= {}\n".format(tag_occurences_dict))
        print("Occurences in Test (Gold standard) = {}\n".format(occurences_dict))
        print("Dictionary of mislabelled tags = {}\n".format(mislabelled_dict))
        print("\n")


if __name__ == "__main__":
    main()
