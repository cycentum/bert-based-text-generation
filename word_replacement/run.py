import sys
from pathlib import Path
import subprocess
import pickle
import fasttext
import numpy as np


def run(inputText, iterSize, modelType="auto", initRandState=None, fileRandState=None, fileFastText=None, **kwargs):
	if modelType=="auto":
		model = fasttext.load_model(str(fileFastText))
		numPred=8
		langCandidate=("__label__en", "__label__ja")
		while True:
			langConfidence=model.predict("".join(inputText), k=numPred)
			langConfidence=dict(zip(*langConfidence))
			contains=False
			for lang in langCandidate:
				if lang in langConfidence:
					contains=True
					break
			if contains: break
			numPred*=2
		for lang in langCandidate:
			if lang not in langConfidence: langConfidence[lang]=0
		conf=np.array([langConfidence[lang] for lang in langCandidate])
		index=conf.argmax()
		lang=langCandidate[index]
		if lang=="__label__en":
			modelType="orig"
		elif lang=="__label__ja":
			modelType="ja-sp"
		print("lang =",lang,", model = ",modelType)
	
	if modelType=="ja-sp":
		DIR_BERT_JA_MODEL=Path("../bert-japanese/model")
		fileInitCheckpoint = DIR_BERT_JA_MODEL/"model.ckpt-1400000"
		fileVocab=DIR_BERT_JA_MODEL/"wiki-ja.vocab"
		fileModel=DIR_BERT_JA_MODEL/"wiki-ja.model"
		kwargs["model_file"]=fileModel
	elif modelType=="orig":
		DIR_BERT_MODEL=Path("../bert_model/cased_L-24_H-1024_A-16")
		fileInitCheckpoint = DIR_BERT_MODEL/"bert_model.ckpt"
		fileVocab=DIR_BERT_MODEL/"vocab.txt"
		fileBertConfig=DIR_BERT_MODEL/"bert_config.json"
		kwargs["bert_config_file"]=fileBertConfig
	
	tmpFiles=[]
	
	argv=[]
	argv.extend(("--model_type", modelType))
	argv.extend(("--init_checkpoint", str(fileInitCheckpoint)))
	argv.extend(("--vocab_file", str(fileVocab)))
	
	if not initRandState is None:
		fileTmpInitRandState=Path(r"./TmpInitRandState.pkl")
		with open(fileTmpInitRandState, "wb") as f: pickle.dump(initRandState, f)
		argv.extend(("--init_rand_state_file", str(fileTmpInitRandState)))
		tmpFiles.append(fileTmpInitRandState)
	if not fileRandState is None:
		argv.extend(("--rand_state_file", str(fileRandState)))
	
	argv.extend(("--iter_size", str(iterSize)))
	
	for key,arg in kwargs.items():
		argv.extend(("--"+key, str(arg)))
	
	fileTmpInput=Path(r"./TmpInput.txt")
	tmpFiles.append(fileTmpInput)
	argv.extend(("--input_file", str(fileTmpInput)))
	with open(fileTmpInput, "w", encoding="utf8") as f:
		for text in inputText: print(text, file=f)
	
	fileTmpOutput=Path(r"./TmpOutput.txt")
	tmpFiles.append(fileTmpOutput)
	argv.extend(("--output_file", str(fileTmpOutput)))
	
	subprocess.run([sys.executable, "./word_replacement.py", *argv])
	
	outputText=[]
	with open(fileTmpOutput, "r", encoding="utf8") as f:
		for line in f:
			outputText.append(line.rstrip())
			
	for f in tmpFiles: f.unlink()
	
	return outputText

if __name__ == "__main__":
	DIR_DATA=Path("../data")
# 	fileInput=DIR_DATA/"InputJ.txt"
	fileInput=DIR_DATA/"InputE.txt"
	inputText=[]
	with open(fileInput, "r", encoding="utf8") as f:
		for line in f:
			inputText.append(line.rstrip())
	
	fileRandState=DIR_DATA/'RandState.pkl'
	iterSize=4
	mask_prob=0.2
	
	fileFastText=DIR_DATA/"lid.176.bin"

	generatedText=run(inputText, iterSize, mask_prob=mask_prob, fileRandState=fileRandState, fileFastText=fileFastText)
	
	for text in generatedText:
		print(text)
