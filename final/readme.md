(1) 實驗環境描述（所需資料、系統、所需所有套件版本<br>
    python 3.5, ubuntu 16.04, 套件使用:<br> 
    keras==2.0.8,<br>
    keras-vis==0.4.1,<br>
    pandas==0.20.3,<br> 
    numpy==1.13.3,<br> 
    scipy==1.0.0,<br> 
    opencv-python==3.3.0.10<br>
    以及HTC之judger_medical套件<br><br>
(2) 如何跑 training<br>
    首先執行preprocessing.py，arguments為htc提供之data路徑<br> 
    python preprocessing.py Data_Entry_2017_v2路徑 train.txt路徑 valid.txt路徑<br><br> 
    接下來執行partition_trainData.py，arguments為htc提供之x光照片資料夾路徑<br> 
    python partition_trainData.py images/路徑<br>
    此步驟要耗時比較久，約五至十分鐘，目的在把traininig image做preprocess後存成.npy檔，以加快訓練時讀檔速度<br><br>
    最後執行train.py<br>
    python train.py<br><br>

(3)如何跑 testing<br>
   執行python test.py即可
   
*助教的python指令若預設為python2，則下command時請下python3 xxxx.py
