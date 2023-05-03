import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

class Consumable:
    def __init__(self, file_location):
        self.file_location = file_location
        self.df = pd.read_csv(self.file_location, delimiter=";").drop("No", axis=1)
        self.columns = self.df.columns.to_list()
        self.food_options = list(self.df["Nama"].values)
        self.n_food = len(self.df)
        self.arange_n = np.arange(self.n_food)

        self.rated_options = []
        self.option_ratings = []

    def create_options(self):
        self.sample = self.df.sample(3)
        self.remaining = self.df.drop(self.sample.index)
        return self.sample, self.remaining
    
    def apply_ratings(self, option_ratings):
        self.option_ratings = option_ratings
        self.sample["Rating"] = self.option_ratings
        return self.sample, self.option_ratings
    
    def recommend(self):
        print(self.sample)
        distances = cdist(self.sample.iloc[:, 1:6], self.remaining.iloc[:, 1:6], metric='euclidean')
        for i, item in self.sample.iterrows():
            rating = item["Rating"]
            if rating > 1:
                distances[:, i] *= 6 - rating
        similar_item_idx = np.argpartition(distances.sum(axis=0), 3)[:3]
        similar_items = self.remaining.iloc[similar_item_idx]

        similarity_values = []
        for i, item in similar_items.iterrows():
            similarity_value = 5 - ((distances[:, i].max() - distances[:, i].min()) / distances[:, i].max()) * 4
            similarity_values.append(similarity_value)
        print(similarity_values)
        return similar_items