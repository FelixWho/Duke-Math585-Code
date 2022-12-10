# https://www.quantconnect.com/terminal/processCache?request=embedded_backtest_c748c67f8083e4148a3b567c381ec2b9.html
from AlgorithmImports import *
import tensorflow.compat.v1 as tf # attempt to disable TF error message
tf.disable_v2_behavior() # attempt to disable TF error message
from AlgorithmImports import *
import numpy as np
import statsmodels.api as sm
import random
from keras import backend as K 
from statsmodels.tsa.stattools import coint, adfuller
from QuantConnect.DataSource import *

#from keras.layers import LSTM, Dense, Dropout
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import gc
from matplotlib import pyplot as plt

from datetime import datetime, timedelta
tf.disable_eager_execution()

import pandas as pd
from copy import deepcopy

class NasdaqCustomColumns(NasdaqDataLink):
    def __init__(self) -> None:
        # Select the column "open interest - change".
        self.ValueColumnName = "open interest - change"

class BasicTemplateAlgorithm(QCAlgorithm):

    def Initialize(self) -> None:

        #1. Required: Backtesting
        self.TrainingPeriod = "OOSB"

        #1. Required: Five years of backtest history
        if self.TrainingPeriod == "IS":
            self.SetStartDate(2017, 1, 1)
            self.SetEndDate(2021, 1, 1)
        if self.TrainingPeriod == "OOSA":
            self.SetStartDate(2022, 1, 1)
            self.SetEndDate(2022, 11, 1)
        if self.TrainingPeriod == "OOSB":
            self.SetStartDate(2016, 1, 1)
            self.SetEndDate(2017, 1, 1)
        if self.TrainingPeriod == "OOSC":
            self.SetStartDate(2010, 1, 1)
            self.SetEndDate(2011, 1, 1)
        #2. Required: Alpha Streams Models:
        self.SetBrokerageModel(BrokerageName.AlphaStreams)
        #3. Required: Significant AUM Capacity
        self.SetCash(10000000)
        #4. Required: Benchmark to SPY, add the equity first
        self.bench = self.AddEquity("SPY", Resolution.Daily)
        self.SetBenchmark("SPY")
        self.lookback = 30
        self.long = "SPY"
        self.short = "SPY"
        self.models = []
        self.SetRiskManagement(MaximumDrawdownPercentPerSecurity(0.05))
        self.SetExecution(ImmediateExecutionModel())

        # self.AddEquity("SPY", Resolution.Daily)
            
        # Initialize the LSTM model
        #self.BuildSPYModel()
        #self.BuildQQQModel()

        #self.AddData(NasdaqCustomColumns, 'FINRA/FNYX_EWO', Resolution.Daily, TimeZones.NewYork)
        #self.sent_EWO = self.SMA('FINRA/FNYX_EWO', 5)

        #self.AddData(NasdaqCustomColumns, 'FINRA/FORF_THNCF', Resolution.Daily, TimeZones.NewYork)
        #self.sent_EWK = self.SMA('FINRA/FORF_THNCF', 5)
        
        # Set Scheduled Event Method For Our Model
        #self.Schedule.On(self.DateRules.EveryDay("SPY"), 
        #    self.TimeRules.BeforeMarketClose("SPY", 10), # 10 min after US market opens, doesn't work?????
        #    self.EveryDayBeforeMarketClose)

        # Set Scheduled Event Method For Our Model Retraining every month
        self.Schedule.On(self.DateRules.MonthStart(), 
            self.TimeRules.At(0, 0), 
            Action(self.BuildAllModels))

        # Set Scheduled Event Method For Our Model
        self.Schedule.On(self.DateRules.EveryDay(), 
            self.TimeRules.BeforeMarketClose("SPY", 5), 
            Action(self.EveryDayBeforeMarketClose))
        
        self.pairs = ["EWD", # Norway and Sweden
                                "EWO", "EWK", # Belgium and Austria
                                "EWJ",  # Japan and South Korea
                                "AAXJ", # iShares MSCI All Country Asia ex Japan Index ETF 2008.8
                                "EWZ", # Chile and Brazil
                                "EWI"]
                    #"EWD",  # iShares MSCI Sweden Index ETF 1996
                    #"EWL",  # iShares MSCI Switzerland Index ETF 1996
                    #"GXC",  # SPDR S&P China ETF 2007.4
                    #"EWC",  # iShares MSCI Canada Index ETF 1996
                    #"EWZ"]
                    #"EWO",
                    #"EWK",
                    #"ECH",
                    #"EGPT"] 
                    #"GAF",  # SPDR S&P Emerging Middle East & Africa ETF 2007.4
                    #"ENZL", # iShares MSCI New Zealand Investable Market Index Fund 2010.9
                    #"NORW",  # Global X FTSE Norway 30 ETF 2011
                    #"EWY",  # iShares MSCI South Korea Index ETF 2000.6
                    #"EWP",  # iShares MSCI Spain Index ETF 1996
                    #"EWD",  # iShares MSCI Sweden Index ETF 1996
                    #"EWL",  # iShares MSCI Switzerland Index ETF 1996
                    #"GXC",  # SPDR S&P China ETF 2007.4
                    #"EWC",  # iShares MSCI Canada Index ETF 1996
                    #"EWZ",  # iShares MSCI Brazil Index ETF 2000.8
                    #"ECH",  # iShares MSCI Chile Investable Market Index ETF 2018  2008
                    #"AND",  # Global X FTSE Andean 40 ETF 2011.3
                    #"AIA",
                    #"EWO",
                    #"EWK",
                    #"ECH",
                    #"EGPT",
                    #"IVV",  # iShares S&P 500 Index 2001
                    #"AAXJ", # iShares MSCI All Country Asia ex Japan Index ETF 2008.8
                    #"EWQ",  # iShares MSCI France Index ETF 2000
                    #"EWH",  # iShares MSCI Hong Kong Index ETF 1999
                    # "EPI",  # WisdomTree India Earnings ETF 2008.3
                    #"EIDO"]  # iShares MSCI Indonesia Investable Market Index ETF 2008.3]  # iShares S&P Asia 50 Index ETF 1996
        '''
                    "EWO",  # iShares MSCI Austria Investable Mkt Index ETF 1996
                    "EWK",  # iShares MSCI Belgium Investable Market Index ETF 1996
                    "ECH",  # iShares MSCI Chile Investable Market Index ETF 2018  2008
                    # "EGPT", # Market Vectors Egypt Index ETF 2011
                    "EWJ",  # iShares MSCI Japan Index ETF 1999
                    "EZU",  # iShares MSCI Eurozone ETF 2000
                    "EWW",  # iShares MSCI Mexico Inv. Mt. Idx 2000
                    # "ERUS", # iShares MSCI Russia ETF 2011
                    "IVV",  # iShares S&P 500 Index 2001
                    "AAXJ", # iShares MSCI All Country Asia ex Japan Index ETF 2008.8
                    "EWQ",  # iShares MSCI France Index ETF 2000
                    "EWH",  # iShares MSCI Hong Kong Index ETF 1999
                    # "EPI",  # WisdomTree India Earnings ETF 2008.3
                    "EIDO",  # iShares MSCI Indonesia Investable Market Index ETF 2008.3
                    "EWI",]  # iShares MSCI Italy Index ETF 1996
                    # "ADRU"] # BLDRS Europe 100 ADR Index ETF 2003
        # self.pairs = ["QQQ", "SPY"]
        '''

        self.nameToIndex = {k: v for v, k in enumerate(self.pairs)}
        self.indexToName = {v: k for v, k in enumerate(self.pairs)}

        # The following ETFs have history() data
        # "NORW", "EWL", "ENZL", "EWY", "EWP", "EWD", "GXC", "EWC", "EWZ", "AIA", "EWO", "EWK", "ECH", "EWJ", "EZU", "EWW", "IVV", "AAXJ", "EWQ", "EWH", "EIDO", "EWI", "ADRU"
        self.symbols = []
        for i in self.pairs:
            self.symbols.append(self.AddEquity(i, Resolution.Daily).Symbol)
            # set our models
        self.bench.SetFeeModel(CustomFeeModel(self))
        self.bench.SetFillModel(CustomFillModel(self))
        self.bench.SetSlippageModel(CustomSlippageModel(self))
        self.bench.SetBuyingPowerModel(CustomBuyingPowerModel(self))

        # Algo Hyperparameters
        self.ibsEntry = 0.5 # Enter trade if abs(ibs) > self.ibsEntry
        self.ibsExit = 0.5

    def calcIBS(self, symbols):
        ibs = []
        for i in range(len(symbols)):
            security = self.Securities[symbols[i]]
            if security.High == security.Low:
                ibs.append(1)
            else:
                ibs.append((security.Close - security.Low) / (security.High - security.Low))
        return ibs

    def calcADFs(self, symbols):
        adfs = []
        for i in range(len(symbols)):
            for j in range(i,len(symbols)):
                adfs.append(self.stats([symbols[i],symbols[j]])[0])
        return adfs

    def stats(self, symbols):
        
        #Use Statsmodels package to compute linear regression and ADF statistics
        self.df = self.History(symbols, self.lookback)
        self.dg = self.df['close'].unstack(level=0)
        
        #self.Debug(self.dg.head())
        
        ticker1= str(symbols[0])
        ticker2= str(symbols[1])

        Y = self.dg[ticker1].apply(lambda x: math.log(x))
        X = self.dg[ticker2].apply(lambda x: math.log(x))
        
        X = sm.add_constant(X)
        model = sm.OLS(Y,X)
        results = model.fit()
        sigma = math.sqrt(results.mse_resid) # standard deviation of the residual
        slope = results.params[1]
        intercept = results.params[0]
        res = results.resid #regression residual mean of res =0 by definition
        zscore = res/sigma
        adf = adfuller (res)
        #self.Debug([adf])
        return [adf, zscore, slope]

    def BuildGeneralModel(self, close):
        qb = self
        ### Preparing Data
        # Get historical data
        
        # self.Debug("Started building model for "+str(security))
        # history = qb.History(security, 252*2, Resolution.Daily)
        # self.Debug(str(security) + ": " + str(history))
        
        # Select the close column and then call the unstack method.
        # close = history['close'].unstack(level=0)
        # self.Debug(close)
        close = close.T

        # Scale data onto [0,1]
        self.scaler = MinMaxScaler(feature_range = (0, 1))
        
        # Transform our data
        df = pd.DataFrame(self.scaler.fit_transform(close), index=close.index)
        
        #self.Debug("Start: " + str(close.index[0]) + ", Close: " + str(close.index[-1]))
        # Feature engineer the data for input.
        input_ = df.iloc[1:]
        
        # Shift the data for 1-step backward as training output result.
        output = df.shift(-1).iloc[:-1]
        
        # Build feauture and label sets (using number of steps 60, and feature rank 1)
        features_set = []
        labels = []
        for i in range(60, input_.shape[0]):
            features_set.append(input_.iloc[i-60:i].values.reshape(-1, 1))
            labels.append(output.iloc[i])
        features_set, labels = np.array(features_set), np.array(labels)
        # self.Debug(features_set.shape)
        features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))
        # self.Debug("Features shape")
        # self.Debug(str(features_set.shape[0]) + "," + str(features_set.shape[1]))
        ### Build Model
        # Build a Sequential keras model
        returnModel = Sequential()
        
        # Add our first LSTM layer - 50 nodes
        returnModel.add(LSTM(units = 50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
        # Add Dropout layer to avoid overfitting
        returnModel.add(Dropout(0.2))
        # Add additional layers
        returnModel.add(LSTM(units=50, return_sequences=True))
        returnModel.add(Dropout(0.2))
        returnModel.add(LSTM(units=50))
        returnModel.add(Dropout(0.2))
        returnModel.add(Dense(units = 5))
        returnModel.add(Dense(units = 1))
        
        # Compile the model. We use Adam as optimizer for adpative step size and MSE as loss function since it is continuous data.
        returnModel.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mae', 'acc'])

        # Set early stopping callback method
        callback = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
        
        # Fit the model to our data, running 20 training epochs
        #self.Debug(datetime.now())
        #returnModel.fit(features_set, labels, epochs = 20, batch_size = 1000, callbacks=[callback])
        history = returnModel.fit(features_set, labels, epochs = 20, batch_size = 1000, callbacks=[callback])
        #self.Debug(history.history)
        #self.Debug(datetime.now())
        #self.Debug("Finished building model")

        return returnModel

    def BuildAllModels(self):
        self.Debug("NEW MONTH -- BUILDING ALL MODELS "+str(self.Time))
        qb = self
        for model in self.models:
            K.clear_session()
            gc.collect
            del model
        temp_models = []

        history = qb.History(qb.symbols, 70, Resolution.Daily)
        close = history['close'].unstack(level=0)

        for securityName in qb.symbols:
            self.Debug("working on " + str(securityName))
            temp_models.append(self.BuildGeneralModel(pd.DataFrame([close[str(securityName)]])))
        
        self.models = temp_models

    def EveryDayBeforeMarketClose(self) -> None:
        #self.Debug("Scheduled event before market close")

        qb = self

        # Use this to make algo decisions
        chart = [[str(sym.Value), -1, -1, -1, -1] for sym in qb.symbols] # symbol name, IBS, predicted price, actual price (respectively)

        # IBS
        ibsCalculations = self.calcIBS(qb.symbols)

        predictedPrices = []

        history = qb.History(qb.symbols, 60, Resolution.Daily)
        if history.empty: return

        close = history['close'].unstack(level=0)

        for idx, securityName in enumerate(qb.symbols):
            # Raw data transform
            unscaledDf = pd.DataFrame([close[str(securityName.Value)]]).T
            scaledDf = pd.DataFrame(self.scaler.transform(unscaledDf), index=unscaledDf.index)

            # Feature engineer the data for input
            input_ = []
            i
