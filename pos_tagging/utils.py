from nltk import word_tokenize


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
    # print(len(tagged_sentences[0]))
    # print(len(test_deserialized[0]))
    # https://stackoverflow.com/questions/43747451/stanford-nlp-tagger-via-nltk-tag-sents-splits-everything-into-chars
    # fixed bug where tagger was reading by char

    # uses our custom trained tagger on the original sentences
    # tagged_sentences = tagger.tag_sents(word_tokenize(sent) for sent in original_sentences)
    # print(tagged_sentences[0])
    count = 0
    for test, tagged in zip(tagged_sentences, test_deserialized):
        if len(test) == 0: continue
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
