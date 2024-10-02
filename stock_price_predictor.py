class StockPricePredictor:
    def __init__(self,csv_file):
        self.dataset=pd.read_csv(csv_file)
        self.X=self.dataset.iloc[:,[1,2,3,5,6]].values
        self.y=self.dataset.iloc[:,self.dataset.columns=="Close"].values
        self.X_train,self.X_test,self.y_train,self.y_test=None,None,None,None
        self.regressor=LinearRegression()
        self.scaler=StandardScaler()

    def preprocess_data(self):
        imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
        self.X=imputer.fit_transform(self.X)
        self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.X,self.y,test_size=0.2,train_size=0.8,random_state=1)
        self.X_train=self.scaler.fit_transform(self.X_train)
        self.X_test=self.scaler.transform(self.X_test)

    def train_model(self):
        self.regressor.fit(self.X_train,self.y_train)

    def predict(self):
        return self.regressor.predict(self.X_test)

    def plot_results(self, y_pred):
        X_test_single_feature=self.X_test[:,0]
        sorted_indices=np.argsort(X_test_single_feature)
        X_test_single_feature_sorted=X_test_single_feature[sorted_indices]
        y_test_sorted=self.y_test[sorted_indices]
        y_pred_sorted=y_pred[sorted_indices]
        plt.scatter(X_test_single_feature_sorted,y_test_sorted,color='red',label='Actual values',s=10)
        plt.plot(X_test_single_feature_sorted,y_pred_sorted,color='blue',label='Predicted values')
        plt.title('Stock Pricing: Actual vs Predicted')
        plt.xlabel('Open Price')
        plt.ylabel('Close Price')
        plt.legend()
        plt.show()

    def run(self):
        self.preprocess_data()
        self.train_model()
        y_pred=self.predict()
        self.plot_results(y_pred)
        
if __name__=="__main__":
    predictor=StockPricePredictor("AAPL.csv")
    predictor.run()
