format:
	# format files to use "#" as seperator
	python clean_data.py A3DataCleaned/Domain1Train.txt A3DataCleaned/Domain1Train.cleaned.txt
	python clean_data.py A3DataCleaned/Domain2Train.txt A3DataCleaned/Domain2Train.cleaned.txt
	python clean_data.py A3DataCleaned/Domain1Test.txt A3DataCleaned/Domain1Test.cleaned.txt
	python clean_data.py A3DataCleaned/Domain2Test.txt A3DataCleaned/Domain2Test.cleaned.txt
	python clean_data.py A3DataCleaned/ELLTest.txt A3DataCleaned/ELLTest.cleaned.txt
	python clean_data.py A3DataCleaned/ELLTrain.txt A3DataCleaned/ELLTrain.cleaned.txt

format-dev:
	# format files to use "#" as seperator
	python clean_data.py A3DataCleaned/Domain1Train.txt A3DataCleaned/Domain1Train.cleaned.txt --split=5000
	python clean_data.py A3DataCleaned/Domain2Train.txt A3DataCleaned/Domain2Train.cleaned.txt --split=5000

train:
	# train model
	java -classpath stanford-postagger/stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTagger -prop Domain1.prop
	java -classpath stanford-postagger/stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTagger -prop Domain2.prop
	java -classpath stanford-postagger/stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTagger -prop ELL.prop

test:
	# test model
	mkdir -p output
	java -classpath stanford-postagger/stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTagger -model Domain1.tagger -testFile A3DataCleaned/Domain1Test.cleaned.txt > output/Domain1Test-tagged.Domain1Train.txt
	java -classpath stanford-postagger/stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTagger -model Domain2.tagger -testFile A3DataCleaned/Domain2Test.cleaned.txt > output/Domain2Test-tagged.Domain2Train.txt
	java -classpath stanford-postagger/stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTagger -model Domain1.tagger -testFile A3DataCleaned/Domain2Test.cleaned.txt > output/Domain2Test-tagged.Domain1Train.txt
	java -classpath stanford-postagger/stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTagger -model Domain2.tagger -testFile A3DataCleaned/Domain1Test.cleaned.txt > output/Domain1Test-tagged.Domain2Train.txt
	java -classpath stanford-postagger/stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTagger -model Domain1.tagger -testFile A3DataCleaned/ELLTest.cleaned.txt > output/ELLTest-tagged.Domain1Train.txt
	java -classpath stanford-postagger/stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTagger -model Domain2.tagger -testFile A3DataCleaned/ELLTest.cleaned.txt > output/ELLTest-tagged.Domain2Train.txt
	java -classpath stanford-postagger/stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTagger -model ELL.tagger -testFile A3DataCleaned/ELLTest.cleaned.txt > output/ELLTest-tagged.ELLTrain.txt

test-hmm:
	python tagger.py --train_file A3DataCleaned/Domain1Train.txt --test_file A3DataCleaned/Domain1Test.txt --tagger hmm
	python tagger.py --train_file A3DataCleaned/Domain1Train.txt --test_file A3DataCleaned/Domain2Test.txt --tagger hmm
	python tagger.py --train_file A3DataCleaned/Domain2Train.txt --test_file A3DataCleaned/Domain2Test.txt --tagger hmm
	python tagger.py --train_file A3DataCleaned/Domain2Train.txt --test_file A3DataCleaned/Domain1Test.txt --tagger hmm
	python tagger.py --train_file A3DataCleaned/Domain1Train.txt --test_file A3DataCleaned/ELLTest.txt --tagger hmm
	python tagger.py --train_file A3DataCleaned/Domain2Train.txt --test_file A3DataCleaned/ELLTest.txt --tagger hmm
	python tagger.py --train_file A3DataCleaned/ELLTrain.txt --test_file A3DataCleaned/ELLTest.txt --tagger hmm

test-brill:
	python tagger.py --train_file A3DataCleaned/Domain1Train.txt --test_file A3DataCleaned/Domain1Test.txt --tagger brill
	python tagger.py --train_file A3DataCleaned/Domain1Train.txt --test_file A3DataCleaned/Domain2Test.txt --tagger brill
	python tagger.py --train_file A3DataCleaned/Domain2Train.txt --test_file A3DataCleaned/Domain2Test.txt --tagger brill
	python tagger.py --train_file A3DataCleaned/Domain2Train.txt --test_file A3DataCleaned/Domain1Test.txt --tagger brill
	python tagger.py --train_file A3DataCleaned/Domain1Train.txt --test_file A3DataCleaned/ELLTest.txt --tagger brill
	python tagger.py --train_file A3DataCleaned/Domain2Train.txt --test_file A3DataCleaned/ELLTest.txt --tagger brill
	python tagger.py --train_file A3DataCleaned/ELLTrain.txt --test_file A3DataCleaned/ELLTest.txt --tagger brill

archive:
	zip -r -X cmput497_a3_yonael_zichun3.zip . -x .\* -x */\__pycache__/\* -x *.pyc -x venv/\* -x .git/\* -x .vscode/\*
