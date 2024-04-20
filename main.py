from load_data import load_data
from train_model import train_model
from models_pipeline import get_pipeline
from sklearn.model_selection import StratifiedKFold

print(99)
data_dir_path = 'ORLDatabase-2021'
over_sampled_dir = 'over_sampled'

X, y = load_data(data_dir_path)

pipeline = get_pipeline()

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_model(pipeline, X, y, kfold)