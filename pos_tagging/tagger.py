import logging

import nltk

from nltk.tbl.template import Template
from nltk.tag.hmm import (
    HiddenMarkovModelTrainer,
    HiddenMarkovModelTagger,
    _identity,
    LidstoneProbDist,
    LazyMap,
    unique_list,
)
from nltk.tag.brill_trainer import BrillTaggerTrainer, BrillTagger
from nltk.tag.brill import Word, Pos
from nltk.tag import UnigramTagger

logger = logging.getLogger("tagger")


class Tagger(object):
    def __init__(self):
        self.trainer = None
        self.tagger = None

    @staticmethod
    def factory(tagger_name):
        if tagger_name == "hmm":
            return HMMTagger()
        elif tagger_name == "brill":
            return BrillTagger()

    def train(self, train_data):
        pass

    def test(self, test_data):
        pass


class HMMTagger(Tagger):
    # https://github.com/nltk/nltk/blob/3.4.5/nltk/tag/hmm.py#L157
    def train(self, labeled_sequence):
        def estimator(fd, bins):
            return LidstoneProbDist(fd, 0.1, bins)

        labeled_sequence = LazyMap(_identity, labeled_sequence)
        symbols = unique_list(word for sent in labeled_sequence for word, tag in sent)
        tag_set = unique_list(tag for sent in labeled_sequence for word, tag in sent)

        trainer = HiddenMarkovModelTrainer(tag_set, symbols)
        hmm = trainer.train_supervised(labeled_sequence, estimator=estimator)
        hmm = HiddenMarkovModelTagger(
            hmm._symbols,
            hmm._states,
            hmm._transitions,
            hmm._outputs,
            hmm._priors,
            transform=_identity,
        )
        self.tagger = hmm

    def test(self, test_data):
        return self.tagger.evaluate(test_data)


class BrillTagger(Tagger):
    def __init__(self):
        pass

    def train(self, data):
        # split 10% of training data for baseline tagger
        cutoff_idx = int(len(data) * 0.1)
        baseline_data = data[:cutoff_idx]
        training_data = data[cutoff_idx:]
        logger.info(
            "Number of sentences, Baseline data: {}, Training data: {}".format(
                len(baseline_data), len(training_data)
            )
        )

        # baseline tagger: unigram tagger
        self.baseline_tagger = UnigramTagger(baseline_data)

        templates = [Template(Pos([-1])), Template(Pos([-1]), Word([0]))]
        self.trainer = BrillTaggerTrainer(self.baseline_tagger, templates=templates)
        self.tagger = self.trainer.train(training_data)

    def test(self, test_data):
        logger.info(
            "Baseline tagger accuracy: {:.2f}%".format(
                self.baseline_tagger.evaluate(test_data) * 100.0
            )
        )
        return self.tagger.evaluate(test_data)
