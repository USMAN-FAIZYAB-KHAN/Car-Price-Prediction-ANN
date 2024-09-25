import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OrdinalEncoder , OneHotEncoder                                     # type: ignore
from sklearn.metrics import accuracy_score, classification_report                                                # type: ignore
from sklearn.model_selection import train_test_split                                                                 # type: ignore
from sklearn.neural_network import MLPClassifier                                                                    # type: ignore
from sklearn.metrics import accuracy_score, classification_report                                                       # type: ignore

df = pd.read_csv('car_data.csv')

# Cleaning
df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce')
df.dropna(inplace=True)

# No of data rows
print("No of data rows", df.shape[0])

# Normalization
scaler = MinMaxScaler()
df[['km_driven', 'mileage(km/ltr/kg)', 'engine']] = scaler.fit_transform(
    df[['km_driven', 'mileage(km/ltr/kg)', 'engine']]
    )

# Standardization
standard_scaler = StandardScaler()
df[['year', 'max_power']] = standard_scaler.fit_transform(df[['year', 'max_power']])

# One Hot Encoding
one_hot_encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")

df = pd.concat([
    df.drop(columns=['fuel', 'seller_type', 'transmission']),
    one_hot_encoder.fit_transform(df[['fuel', 'seller_type', 'transmission']])
], axis=1)

# Ordinal Encoding
ordinal_encoder = OrdinalEncoder()
df['owner'] = ordinal_encoder.fit_transform(df[['owner']])
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OrdinalEncoder , OneHotEncoder                                     # type: ignore


# Binning
df['price_category'] = pd.qcut(df['selling_price'], q=3, labels=False)

# Scaling
df['seats'] = scaler.fit_transform(df[['seats']])

df.to_csv('processed_car_data.csv', index=False)

X = df.drop(['name', 'selling_price', 'price_category'], axis=1)
y = df['price_category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Number of training rows", X_train.shape[0])
print("Number of testing rows", X_test.shape[0])

mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)

mlp_classifier.fit(X_train, y_train)

y_pred = mlp_classifier.predict(X_test)

print("Hidden layer sizes:", mlp_classifier.hidden_layer_sizes)
print("Number of layers:", mlp_classifier.n_layers_)
print("Number of iterations:", mlp_classifier.n_iter_)
print("Classes:", mlp_classifier.classes_)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
