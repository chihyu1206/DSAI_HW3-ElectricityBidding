import pandas as pd
import argparse
import numpy as np
from sklearn.svm import SVR
from datetime import datetime, timedelta
def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()

def output(path, data):
    # import pandas as pd

    df = pd.DataFrame(data, columns=["time", "action", "target_price", "target_volume"])
    df.to_csv(path, index=False)

    return

def create_dataset(dataset, look_back=1):
	dataX, dataY = list(), list()
	for i in range(len(dataset) - look_back - 1):
		data = dataset[i : (i + look_back)]
		dataX.append(data)
		dataY.append(dataset[i + look_back])
	return np.array(dataX), np.array(dataY)
	
if __name__ == "__main__":
    args = config()
    target_df = pd.read_csv("./sample_data/target23.csv")    
    dataset = target_df.drop(columns=["time"]).values
    
    dataGen = dataset[:, 0]
    dataCon = dataset[:, 1]
    epsilon = 1e-4
    minGenValue, maxGenValue = dataGen.min() + epsilon, dataGen.max()
    minConValue, maxConValue = dataCon.min() + epsilon, dataCon.max()
    

    dataGen = (dataGen - minGenValue) / (maxGenValue - minGenValue)
    dataCon = (dataCon - minConValue) / (maxConValue - minConValue)
    
    look_back = 24
    X_train_gen, y_train_gen = create_dataset(dataGen, look_back)
    X_train_con, y_train_con = create_dataset(dataCon, look_back)
    y_train_gen = np.reshape(y_train_gen, (y_train_gen.shape[0]))
    y_train_con = np.reshape(y_train_con, (y_train_con.shape[0]))    
    
    gen_svr = SVR(kernel="poly", C=100, gamma="auto", degree=6, epsilon=0.1, coef0=1)
    con_svr = SVR(kernel="poly", C=100, gamma="auto", degree=6, epsilon=0.1, coef0=1)
    gen_svr.fit(X_train_gen, y_train_gen)
    con_svr.fit(X_train_con, y_train_con)
    
    # Prediction
    con_df = pd.read_csv(args.consumption)
    gen_df = pd.read_csv(args.generation)
    
    LASTTIME = con_df.iloc[-1, 0]
    lastTime = datetime.strptime(LASTTIME, "%Y-%m-%d %H:%M:%S")
    
    con_data = con_df.drop(columns=["time"]).values
    gen_data = gen_df.drop(columns=["time"]).values
    
    scaled_con = (con_data - minConValue) / (maxConValue - minConValue)
    scaled_gen = (gen_data - minGenValue) / (maxGenValue - minGenValue)
    
    train_con = scaled_con[-24:]
    train_gen = scaled_gen[-24:]
    train_con = train_con.reshape(1, 24)
    train_gen = train_gen.reshape(1, 24)
    outputs=list()
    for i in range(24):

        lastTime = lastTime + timedelta(hours=1)
        predGenValue = gen_svr.predict(train_gen)
        predConValue = con_svr.predict(train_con)
        predGenValue = predGenValue[0]
        predConValue = predConValue[0]
        
        realGenValue = predGenValue * (maxGenValue - minGenValue) + minGenValue + epsilon
        realConValue = predConValue * (maxConValue - minConValue) + minConValue + epsilon
        
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
        train_con = list(train_con[0])
        train_con.append(predConValue)
        train_con = np.array(train_con[1:])
        train_con = train_con.reshape(1, 24)
        train_gen = list(train_gen[0])
        train_gen.append(predGenValue)
        train_gen = np.array(train_gen[1:])
        train_gen = train_gen.reshape(1, 24)
    
    output(args.output, outputs)
        
         
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
