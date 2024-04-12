from load_data import  load_data
from sklearn.model_selection import train_test_split

data_dir_path = 'ORLDatabase-2021'

X, y = load_data(data_dir_path)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
