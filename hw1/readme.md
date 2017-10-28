Training code 使用方法(與助教規定之hw1_rnn.sh參數相同):
python model_rnn.py data/ output.csv路徑
python model_cnn.py data/ output.csv路徑
python model_best.py data/ output.csv路徑
***
使用套件:
pandas
numpy
keras
sklearn
pickle
***
由於我最好的分數為三個不同model做ensemble
在本機上./hw1_best.sh可以在十分鐘內預測完三個model之結果
做完ensemble之後輸出csv檔
但我不確定在助教電腦上會不會不小心超過時限，故在此做說明。
