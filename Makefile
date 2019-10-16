bootstrap:
	curl https://nlp.stanford.edu/software/stanford-postagger-full-2018-10-16.zip -o stanford-postagger.zip
	unzip stanford-postagger.zip
	mv stanford-postagger-full-2018-10-16 stanford-postagger

format:
	# format files to use "#" as seperator
	python clean_data.py A3DataCleaned/Domain1Train.txt A3DataCleaned/Domain1Train.cleaned.txt
	python clean_data.py A3DataCleaned/Domain2Train.txt A3DataCleaned/Domain2Train.cleaned.txt
	python clean_data.py A3DataCleaned/Domain1Test.txt A3DataCleaned/Domain1Test.cleaned.txt
	python clean_data.py A3DataCleaned/Domain2Test.txt A3DataCleaned/Domain2Test.cleaned.txt

format-dev:
	# format files to use "#" as seperator
	python clean_data.py A3DataCleaned/Domain1Train.txt A3DataCleaned/Domain1Train.cleaned.txt --split=5000
	python clean_data.py A3DataCleaned/Domain2Train.txt A3DataCleaned/Domain2Train.cleaned.txt --split=5000

train:
	# train model
	java -classpath stanford-postagger/stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTagger -prop Domain1.prop
	java -classpath stanford-postagger/stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTagger -prop Domain2.prop

test:
	# test model
	java -classpath stanford-postagger/stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTagger -model Domain1.tagger -testFile A3DataCleaned/Domain1Test.cleaned.txt
	java -classpath stanford-postagger/stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTagger -model Domain2.tagger -testFile A3DataCleaned/Domain2Test.cleaned.txt
