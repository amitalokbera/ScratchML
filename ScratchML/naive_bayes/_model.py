import numpy as np 

class GaussianNB:
    def __init__(self, prior:dict = dict()) -> None:
        self.X = None
        self.y = None 
        self.classes = None 
        self.n_classes = None 
        self._trained = False
        self.class_log_prior = None  
        self.mean = dict()
        self.std = dict()
        self.prior = prior

    def _cal_likelihood(self, data, c):
        epilson = 1e-5
        result = 0
        for idx, val in enumerate(data):
            key = f"{idx}_{c}"
            coeff = 1.0 / np.sqrt(2.0 * np.pi * (self.std[key])**2 + epilson)
            exponent = np.exp(-(np.power(val - self.mean[key], 2) / (2 * (self.std[key])**2 + epilson)))
            result += coeff*exponent
        return result
        
    def fit(self, X,y):
        self.X = X 
        self.y = y 
        self.classes = np.unique(self.y)
        self.n_classes = len(self.classes)
        self.cal_prior = True
        if len(self.prior) == 0:
            self.cal_prior = False
        else:
            for key,val in self.prior.items():
                self.prior[key] = np.log(val)
        for c in self.classes:
            if self.cal_prior:
                self.prior[c] = np.log(len(np.where(self.y == c))/len(self.y))
            x_temp = self.X[np.where(self.y == c)]
            for idx in range(len(x_temp[0])):
                self.mean[f"{idx}_{c}"] = np.mean(x_temp[:,idx])
                self.std[f"{idx}_{c}"] = np.std(x_temp[:,idx])
        self.class_log_prior = self.prior
        self._trained = True 


    def predict(self, Xq):
        if self._trained:
            self.predicted_val = dict()
            for c in self.classes:
                self.predicted_val[c] = self.prior[c] + self._cal_likelihood(Xq,c)
            return sorted(self.predicted_val.items(), key=lambda item: item[1], reverse=True)[0][0]
        else:
            raise Exception("Fit the model, before prediction")

class CategoricalNB:
    def __init__(self, alpha:float = 1.0, prior:dict = dict()) -> None:
        self.prior = prior
        self.alpha = alpha
        self.X = None
        self.y = None 
        self.classes = None 
        self.n_classes = None 
        self.proabilities = dict()
        self.k_count = dict()
        self._trained = False 

    def fit(self, X,y):
        self.X = X 
        self.y = y 
        self.classes = np.unique(self.y)
        self.n_classes = len(self.classes)
        self.cal_prior = True
        if len(self.prior) > 0:
            self.cal_prior = False 
        for c in self.classes:
            self.prior[c] = len(np.where(self.y == c))/len(self.y)
            for col_idx in range(len(self.X[0])):
                col_temp = self.X[:,col_idx]
                col_temp = col_temp[np.where(self.y == c)]
                unique, counts = np.unique(col_temp, return_counts=True)
                counts_data = dict(zip(unique, counts))
                self.k_count[col_idx] = len(np.unique(col_temp))
                for data,count in counts_data.items():
                    key = f"{data}_{col_idx}_{c}"
                    self.proabilities[key] = (len(col_temp[np.where(col_temp == data)]) + self.alpha)/(len(col_temp) + (self.alpha*self.k_count[col_idx]))
        self._trained = True        

    def predict(self, Xq):
        if self._trained:
            self.result = []
            for row in Xq:
                self.pred = dict([(x,0) for x in self.classes])
                for col, val in enumerate(row):
                    for c in self.classes:
                        key = f"{val}_{col}_{c}"
                        if key in self.proabilities:
                            self.pred[c] += np.log(self.proabilities[key])
                        else:
                            self.pred[c] = np.log(self.alpha/(self.alpha*self.k_count[col]))
                        self.pred[c] += self.prior[c]
                self.result.append(sorted(self.pred.items(), key=lambda item: item[1], reverse=True)[0][0])
            return self.result
        else:
            raise Exception("Fit the model, before prediction")