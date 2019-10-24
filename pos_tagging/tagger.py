import logging

import nltk
import nltk.tag.brill as brill

from nltk.tbl.template import Template
from nltk.tag.hmm import (
    HiddenMarkovModelTrainer,
    HiddenMarkovModelTagger,
    _identity,
    LidstoneProbDist,
    LazyMap,
    unique_list,
)
from nltk.tag.stanford import StanfordPOSTagger
from nltk.tag.brill_trainer import BrillTaggerTrainer, BrillTagger
from nltk.tag.brill import Word, Pos
from nltk.tag import UnigramTagger

logger = logging.getLogger("tagger")


class Tagger(object):
    def __init__(self):
        self.trainer = None
        self.tagger = None

    @staticmethod
    def factory(tagger_name, **kwargs):
        if tagger_name == "hmm":
            return HMMTagger()
        elif tagger_name == "brill":
            return BrillTagger(**kwargs)

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
    def train(self, data):
        # baseline tagger: unigram tagger
        hmm = HMMTagger()
        hmm.train(data)
        self.baseline_tagger = hmm.tagger

        # train brill tagger with HMM as baseline
        templates = [
            brill.Template(brill.Pos([-1])),
            brill.Template(brill.Pos([1])),
            brill.Template(brill.Pos([-2])),
            brill.Template(brill.Pos([2])),
            brill.Template(brill.Pos([-2, -1])),
            brill.Template(brill.Pos([1, 2])),
            brill.Template(brill.Pos([-3, -2, -1])),
            brill.Template(brill.Pos([1, 2, 3])),
            brill.Template(brill.Pos([-1]), brill.Pos([1])),
            brill.Template(brill.Word([-1])),
            brill.Template(brill.Word([1])),
            brill.Template(brill.Word([-2])),
            brill.Template(brill.Word([2])),
            brill.Template(brill.Word([-2, -1])),
            brill.Template(brill.Word([1, 2])),
            brill.Template(brill.Word([-3, -2, -1])),
            brill.Template(brill.Word([1, 2, 3])),
            brill.Template(brill.Word([-1]), brill.Word([1])),
        ]
        self.trainer = BrillTaggerTrainer(self.baseline_tagger, templates=templates)
        self.tagger = self.trainer.train(data)

    def test(self, test_data):
        logger.info(
            "Baseline tagger accuracy: {:.2f}%".format(
                self.baseline_tagger.evaluate(test_data) * 100.0
            )
        )
        return self.tagger.evaluate(test_data)
