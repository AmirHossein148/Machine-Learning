
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Preprocessor : 
    def __init__ (self, df):
        self.df = df.copy()
        
    def encode_values (self) :
        encoder = LabelEncoder()
        self.df['Gender'] = encoder.fit_transform(self.df['Gender'])

    def income_transformer (self) :
        self.df['log_income'] = np.log(self.df['Annual Income (k$)'])

    def drop_columns(self):
        self.df.drop(['CustomerID'],axis=1,inplace=True)
      
    def transform (self) : 
        self.encode_values()
        self.drop_columns()
        self.income_transformer()
        return self.df
