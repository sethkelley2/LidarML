from utils import visualize_preds
import pickle

with open('stacked_preds.pkl','rb') as file:
    data = pickle.load(file)

visualize_preds(data)