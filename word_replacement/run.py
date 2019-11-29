import sys
from pathlib import Path
import subprocess
import pickle

def runJa(DIR_BERT_JA_MODEL, DIR_DATA):
	modelType="ja-sp"
	
	fileInitCheckpoint = DIR_BERT_JA_MODEL/"model.ckpt-1400000"
	fileVocab=DIR_BERT_JA_MODEL/"wiki-ja.vocab"
	fileModel=DIR_BERT_JA_MODEL/"wiki-ja.model"
	
	fileInput=DIR_DATA/"InputJ.txt"
	inputText=[]
	with open(fileInput, "r", encoding="utf8") as f:
		for line in f:
			inputText.append(line.rstrip())
			
	iterSize=4
	
	fileRandState=DIR_DATA/"RandJ.pkl"
	
	outputText=run(modelType, inputText, iterSize, fileInitCheckpoint, fileVocab, fileRandState=fileRandState, model_file=fileModel, mask_prob=0.2)
	for text in outputText: print(text)


def runEn(DIR_BERT_MODEL, DIR_DATA):
	modelType="orig"
	
	fileInitCheckpoint = DIR_BERT_MODEL/"bert_model.ckpt"
	fileVocab=DIR_BERT_MODEL/"vocab.txt"
	fileBertConfig=DIR_BERT_MODEL/"bert_config.json"
	
	fileInput=DIR_DATA/"InputE.txt"
	inputText=[]
	with open(fileInput, "r", encoding="utf8") as f:
		for line in f:
			inputText.append(line.rstrip())
	
	iterSize=4
	
	fileRandState=DIR_DATA/"RandE.pkl"
	
	outputText=run(modelType, inputText, iterSize, fileInitCheckpoint, fileVocab, fileRandState=fileRandState, bert_config_file=fileBertConfig)
	for text in outputText: print(text)


def run(modelType, inputText, iterSize, fileInitCheckpoint, fileVocab, initRandState=None, fileRandState=None, **kwargs):
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
	DIR_BERT_JA_MODEL=Path("../bert-japanese/model")
	DIR_BERT_MODEL=Path("../bert_model/cased_L-24_H-1024_A-16")
	DIR_DATA=Path("../data")
	
# 	runJa(DIR_BERT_JA_MODEL, DIR_DATA)
	runEn(DIR_BERT_MODEL, DIR_DATA)
