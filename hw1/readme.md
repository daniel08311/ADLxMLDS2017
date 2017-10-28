Training code 使用方法(與助教規定之hw1_rnn.sh參數相同): <br>
python model_rnn.py data/ output.csv路徑   <br>
python model_cnn.py data/ output.csv路徑  <br>
python model_best.py data/ output.csv路徑  <br>
***
使用套件:
pandas
numpy
keras
sklearn
pickle
***
由於我最好的分數為三個不同model做ensemble<br>
在本機上./hw1_best.sh可以在十分鐘內預測完三個model之結果<br>
做完ensemble之後輸出csv檔<br>
但我不確定在助教電腦上會不會不小心超過時限，故在此做說明。<br>
model_best.py裏頭是我做出來最好的RNN模型(7.64分)<br>
另外兩個拿來ensemble的模型並沒有放在model_best.py裡面(分數都比model_best稍差)<br><br>

而我cnn的model由於一開始是用keras(theano backend)訓練出來的<br>
起初測試可以用tensorflow backend執行就沒多注意<br>
然而最後一天才發現預測結果會壞掉(丟上kaggle從原本7分變成27分)<br>
因此我後來又用tensorflow訓練新的weight<br>
只是本機上tensorflow-gpu一直裝不起來<br>
只好用cpu訓練<br>
時間只容許訓練2個epoch就存成新的weight<br>
因此CNN模型助教跑出來後丟kaggle驗證應該只有16分左右<br>
而best_model則沒問題，可以reproduce出7分<br>
