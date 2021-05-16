# DSAI_HW3-ElectricityBidding
NCKU DSAI HW3 Electricity Bidding


[usage]
1. $ python3 -m pip install pipenv
2. $ git clone https://github.com/chihyu1206/DSAI_HW3-ElectricityBidding.git
3. $ cd DSAI_HW3-ElectricityBidding
4. $ pipenv shell
5. $ pipenv run python main.py
## Description
使用target23.csv之原始資料，搭配Support Vector Regression(SVR)分別對consumption和generation做polynomial regression，最後使用擬合好的模型預測，
若是產電大於用電超過某個額度就賣電，反之則買電。

### Data Analysis
由np.min()會發現資料中有0，做Normalization時要小心，可以再加上一個小數之偏移量來避免錯誤。

### Feature Engineering
如上所述，這裡選擇epsilon == 1e-4，對結果影響不大
```
    target_df = pd.read_csv("./sample_data/target23.csv")    
    dataset = target_df.drop(columns=["time"]).values
    
    dataGen = dataset[:, 0]
    dataCon = dataset[:, 1]
    epsilon = 1e-4
    minGenValue, maxGenValue = dataGen.min() + epsilon, dataGen.max()
    minConValue, maxConValue = dataCon.min() + epsilon, dataCon.max()
```
接著對numpy.ndarray做MinMax Normalization
```
    dataGen = (dataGen - minGenValue) / (maxGenValue - minGenValue)
    dataCon = (dataCon - minConValue) / (maxConValue - minConValue)
```
再來選擇由過去24hr之數據來預測新值
```
    look_back = 24
    X_train_gen, y_train_gen = create_dataset(dataGen, look_back)
    X_train_con, y_train_con = create_dataset(dataCon, look_back)
    y_train_gen = np.reshape(y_train_gen, (y_train_gen.shape[0]))
    y_train_con = np.reshape(y_train_con, (y_train_con.shape[0]))   
```
### Trading Strategy
假設產用電相差大於1就投標 買大於賣選sell 賣大於買選buy
因為不太確定價格走勢 就以略高於市價電費的固定價格來買低(2.75)賣高(3.25)
```
        if realGenValue - realConValue > 1.0:
            _output = list([lastTime.strftime("%Y-%m-%d %H:%M:%S"),
                           "sell",
                           3.25,
                           round(realGenValue-realConValue, 2)])
            outputs.append(_output)
        elif realConValue - realGenValue > 1.0:
            _output = list([lastTime.strftime("%Y-%m-%d %H:%M:%S"),
                           "buy",
                           2.75,
                           round(realConValue-realGenValue, 2)])
            outputs.append(_output)
        else:
            pass
```

## Training
使用Scikit-Learn SVM中的SVR分別對產電/用電做迴歸產生兩個model 這次因為訓練數據較少 容易overfitting 就沒太多著墨在參數之調整
```
    gen_svr = SVR(kernel="poly", C=100, gamma="auto", degree=6, epsilon=0.1, coef0=1)
    con_svr = SVR(kernel="poly", C=100, gamma="auto", degree=6, epsilon=0.1, coef0=1)
    gen_svr.fit(X_train_gen, y_train_gen)
    con_svr.fit(X_train_con, y_train_con)
```


得到預測結果後 輸出到output.csv
```
    output(args.output, outputs)
```
