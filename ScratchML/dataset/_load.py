from calendar import c
import pandas as pd 
import numpy as np 

class Dataset:
    def __init__(self, name:str = "tennis", encoded:bool = False) -> None:
        """
        pass name = 'tennis' to load play tennis dataset, 
        name = 'iris' to load play iris dataset
        """
        self.name = name 
        self.encoded = encoded
    
    def getData(self):
        if self.name == "tennis":
            self.df = pd.read_csv("ScratchML/dataset/tennis.csv")
            if self.encoded:
                for col in self.df.columns:
                    self.df[col] = self.df[col].astype('category')
                    self.df[col] = self.df[col].cat.codes
                return self.df.to_numpy()[:,:-1], np.squeeze(self.df.to_numpy()[:,-1:])
            else:
                return self.df.to_numpy()[:,:-1], np.squeeze(self.df.to_numpy()[:,-1:])
        elif self.name == "iris":
            self.df = pd.read_csv("ScratchML/dataset/iris.csv")
            if self.encoded:
                for col in self.df.columns:
                    if col != 'variety':
                        mean_val, std_val = np.mean(self.df[col].values), np.std(self.df[col].values)
                        self.df[col] = (self.df[col] - mean_val)/std_val
                    else:
                        self.df[col] = self.df[col].astype('category')
                        self.df[col] = self.df[col].cat.codes
                return self.df.to_numpy()[:,:-1], np.squeeze(self.df.to_numpy()[:,-1:])
            else:
                return self.df.to_numpy()[:,:-1], np.squeeze(self.df.to_numpy()[:,-1:])
