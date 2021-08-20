import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

df = pd.read_csv(r"C:\Users\kjay1\Downloads\Used Car price\train-data.csv")

df = df.drop(columns='Unnamed: 0', axis=1)
df = df.drop(columns='New_Price', axis=1)
df = df.drop(columns='Name', axis=1)
df = df.dropna(axis=0, how='any')
df = df.reset_index(drop=True)
nulldrops = df.loc[df['Power'] == 'null bhp'].index
df = df.drop(nulldrops).reset_index(drop=True)

for i in range(len(df["Power"])):
    df.at[i, "Power"] = df["Power"][i].split(" ")[0]

for i in range(len(df["Engine"])):
    df.at[i, "Engine"] = df["Engine"][i].split(" ")[0]

for i in range(len(df["Mileage"])):
    df.at[i, "Mileage"] = df["Mileage"][i].split(" ")[0]


df['Engine'] = df['Engine'].astype('int64')
df['Power'] = df['Power'].astype('float64')
df['Mileage'] = df['Mileage'].astype('float64')
df['Location'] = df['Location'].astype('category')
df['Fuel_Type'] = df['Fuel_Type'].astype('category')
df['Transmission'] = df['Transmission'].astype('category')
df['Owner_Type'] = df['Owner_Type'].astype('category')


X = df.drop(columns=['Price'], axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


hot_enc = OneHotEncoder(sparse=False)
scale = StandardScaler()

Preprocessor = ColumnTransformer(remainder='passthrough',
                                 transformers=[
                                            ("ohe", hot_enc, ['Location', 'Fuel_Type', 'Transmission', 'Owner_Type']),
                                            ("sc", scale, ['Year', 'Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats'])
                                 ])
Preprocessor.fit_transform(X_train)

Base_Model = Pipeline(steps=[
                        ('Preprocessor', Preprocessor),
                        ('Regressor', RandomForestRegressor())
                        ])

Base_Model.fit(X_train, y_train)
score = cross_val_score(Base_Model, X_train, y_train, cv=5, scoring='r2')

Score = np.mean(score)*100
print(f"Accuracy : {Score}%")

pickle.dump(Base_Model, open('Base_Model.pkl', 'wb'))

