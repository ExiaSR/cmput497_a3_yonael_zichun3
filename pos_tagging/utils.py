from nltk import word_tokenize
from prettytable import PrettyTable
from collections import Counter


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
def analyze_mistagged(tagged_sentences, test_deserialized):
    mislabelled = {}
    # https://stackoverflow.com/questions/43747451/stanford-nlp-tagger-via-nltk-tag-sents-splits-everything-into-chars
    # fixed bug where tagger was reading by char

    # uses our custom trained tagger on the original sentences
    wrong_tags_count = 0
    total_tags_count = 0
    wrong_sentences_count = 0
    total_sentences_count = len(test_deserialized)
    for test_sentence, tagged_sentence in zip(tagged_sentences, test_deserialized):
        if len(test_sentence) == 0:
            continue  # skip empty line

        for a, b in zip(test_sentence, tagged_sentence):
            if a[0] != b[0]:
                raise Exception("Something went wrong...")

        # get mislabed tokens
        mislabed_pairs = [
            (test_token[1], tagged_token[1])
            for test_token, tagged_token in zip(test_sentence, tagged_sentence)
            if test_token != tagged_token
        ]

        for each in mislabed_pairs:
            if each in mislabelled:
                mislabelled[each] += 1
            else:
                mislabelled[each] = 1

        wrong_tags_count += len(mislabed_pairs)
        total_tags_count += len(test_sentence)
        wrong_sentences_count += 1 if len(mislabed_pairs) > 0 else 0

    right_tags_count = total_tags_count - wrong_tags_count
    right_sentences_count = total_sentences_count - wrong_sentences_count
    print(
        "Total tags right: {} ({:.4f}%); wrong: {} ({:.4f}%).".format(
            right_tags_count,
            right_tags_count / total_tags_count * 100.0,
            wrong_tags_count,
            wrong_tags_count / total_tags_count * 100.0,
        )
    )
    print(
        "Total sentences right: {} ({:.4f}%); wrong: {} ({:.4f}%).".format(
            right_sentences_count,
            right_sentences_count / total_sentences_count * 100.0,
            wrong_sentences_count,
            wrong_sentences_count / total_sentences_count * 100.0,
        )
    )
    return mislabelled, tagged_sentences


def tag_list(tagged_sents):
    return [tag for sent in tagged_sents for (word, tag) in sent]


# taken from https://stackoverflow.com/a/23715286/4557739
def precesion_and_recall(labels, cm):
    # precision and recall
    true_positives = Counter()
    false_negatives = Counter()
    false_positives = Counter()

    for i in labels:
        for j in labels:
            if i == j:
                true_positives[i] += cm[i, j]
            else:
                false_negatives[i] += cm[i, j]
                false_positives[j] += cm[i, j]

    results = []
    table = PrettyTable()
    table.field_names = ["label", "precision", "recall", "f-score"]
    for each in sorted(labels):
        if true_positives[each] == 0:
            fscore = 0
            results.append({"label": each, "f_score": fscore})
            table.add_row([each, None, None, fscore])
        else:
            precision = true_positives[each] / float(true_positives[each] + false_positives[each])
            recall = true_positives[each] / float(true_positives[each] + false_negatives[each])
            fscore = 2 * (precision * recall) / float(precision + recall)
            results.append(
                {"label": each, "precision": precision, "recall": recall, "f_score": fscore}
            )
            table.add_row([each, precision, recall, fscore])
    return results, table
