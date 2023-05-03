import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

'''
I have a table of 15 food items with 5 taste-related features. 
The features are integers ranging from 1 to 5. 
It is stored in food_data.csv. 
Make a python program using pandas to randomly pick 3 items from the 
dataset and asks the user to rate on a
 scale of 1-5 on how much they like the 3 food items. 
 Then generate 3 different most similar food items from the 
 dataset that the user will most likely like. 
'''

empty_df = pd.DataFrame({"Makanan":[], "Nilai":[]})
empty_df_html = empty_df.to_html(classes="table table-striped", index=False)

def lin_map(i_min, i_max, f_min, f_max, value):
    d1 = value - i_min
    d2 = f_max - f_min
    d3 = i_max - i_min
    return d1 * d2 / d3 + f_min

def recommend_food(file_location, rated_indices, ratings):
    df = pd.read_csv(file_location)
    df['Similarity'] = 0
    for i, row in df.iterrows():
        food_item = row['Food Item']
        similarity = 0
        for feature in ['Sweetness', 'Saltiness', 'Spiciness', 'Sourness', 'Umami']:
            similarity += abs(row[feature] - ratings[rated_indices.index(i)])
        df.loc[i, 'Similarity'] = similarity
    
    # Recommend three other food items based on similarity
    recommendations = df.sort_values(by='Similarity').iloc[:3][['Food Item', 'Similarity']]
    
    # Map the similarity scores to a range of 1-5
    max_score = recommendations['Similarity'].max()
    min_score = recommendations['Similarity'].min()
    recommendations['Similarity'] = recommendations['Similarity'].apply(
        lambda x: 1 + (5 - 1) * (max_score - x) / (max_score - min_score)
    )
    
    return recommendations

class Consumable:
    def __init__(self, file_location, n_input, n_output):
        self.file_location = file_location
        self.n_input = n_input
        self.n_output = n_output
        self.dataset = pd.read_csv(self.file_location, delimiter=";")
        self.columns = self.dataset.columns.to_list()
        self.food_options = list(self.dataset["Nama"].values)
        self.n_food = len(self.dataset)
        self.arange_n = np.arange(self.n_food)
        self.selected_options = []

        self.scaler = MinMaxScaler()
        self.scaled_ratings = self.scaler.fit_transform(self.dataset.iloc[:,2:])
        self.similarity = cosine_similarity(self.scaled_ratings)
        self.rating_options_df: pd.DataFrame
        self.rating_options_list: list

        self.rating_vector: np.ndarray

    # def generate_options_remaining(self):
    #     self.options = np.random.choice(
    #         self.arange_n, 
    #         size=self.n_input, 
    #         replace=False)
    #     remaining = np.setdiff1d(np.arange(self.n_food), self.options)
    #     rating_options = self.dataset.loc[self.options]["Nama"]
        
    #     self.rating_options_df = pd.DataFrame(rating_options)
    #     self.rating_options_list = list(rating_options)
    #     output = pd.DataFrame(rating_options)
    #     # print(output)
    #     return output
    
    def generate_options_remaining(self):
        remaining = np.setdiff1d(self.arange_n, self.selected_options)
        self.options = np.random.choice(
            remaining, 
            size=self.n_input, 
            replace=False)
        rating_options = self.dataset.loc[self.options]["Nama"]
        
        self.rating_options_df = pd.DataFrame(rating_options)
        self.rating_options_list = list(rating_options)
        output = pd.DataFrame(rating_options)
        # print(output)
        return output
    
    def get_user_ratings(self, ratings):
        self.rating_options_df["Rating"] = ratings
        return self.rating_options_df
    
    def generate_rating_vector(self):
        rating_vector = np.zeros(self.n_food)
        for idx, row in self.rating_options_df.iterrows():
            rating_vector[idx] = row["Rating"]
        self.rating_vector = rating_vector
        return rating_vector
    
    def generate_similarity(self):
        a = self.similarity.dot(self.rating_vector)
        b = np.linalg.norm(self.similarity, axis=1)
        c = np.linalg.norm(self.rating_vector)
        input_similarity = a / (b * c)
        mapped = lin_map(0, 1, 1, 5, input_similarity)
        self.input_similarity = mapped
        return input_similarity
    
    def recommend(self):
        # Exclude selected options from food_options list
        remaining_options = list(set(self.food_options) - set(self.rating_options_list) - set(self.selected_options))
        food_recommendation_indices = [self.food_options.index(option) for option in remaining_options]
        food_recommendation_similarity = self.input_similarity[food_recommendation_indices]
        food_recommendations = [self.food_options[i] for i in food_recommendation_indices]
        food_recommendation_df = pd.DataFrame({"Makanan":food_recommendations, "Score":food_recommendation_similarity})
        final_recommendation = food_recommendation_df[:self.n_output]
        final_recommendation["Nilai Keyakinan (1-5)"] = list(np.round(final_recommendation["Score"], 2))
        return final_recommendation

    def sequence(self, ratings):
        self.generate_options_remaining()
        self.get_user_ratings(ratings)
        self.generate

    # def get_user_ratings(self, ratings):
    #     self.rating_options_df["Rating"] = ratings
    #     return self.rating_options_df
    
    # def generate_rating_vector(self):
    #     rating_vector = np.zeros(self.n_food)
    #     for idx, row in self.rating_options_df.iterrows():
    #         rating_vector[idx] = row["Rating"]
    #     self.rating_vector = rating_vector
    #     return rating_vector
    
    # def generate_similarity(self):
    #     a = self.similarity.dot(self.rating_vector)
    #     b = np.linalg.norm(self.similarity, axis=1)
    #     c = np.linalg.norm(self.rating_vector)
    #     input_similarity = a / (b * c)
    #     mapped = lin_map(0, 1, 1, 5, input_similarity)
    #     self.input_similarity = mapped
    #     return input_similarity
    
    # def recommend(self):
    #     food_recommendation_indices = np.argsort(self.input_similarity)[::-1]
    #     food_recommendation_similarity = self.input_similarity[food_recommendation_indices]
    #     food_recommendations = [self.food_options[i] for i in food_recommendation_indices]
    #     food_recommendation_df = pd.DataFrame({"Makanan":food_recommendations, "Score":food_recommendation_similarity})
    #     mask = food_recommendation_df["Makanan"].isin(self.rating_options_list)
    #     food_recommendation_filter = food_recommendation_df[~mask]
    #     final_recommendation = food_recommendation_filter[:self.n_output]
    #     final_recommendation["Nilai Keyakinan (1-5)"] = list(np.round(final_recommendation["Score"], 2))
    #     return final_recommendation

    # def sequence(self, ratings):
    #     self.generate_options_remaining()
    #     self.get_user_ratings(ratings)
    #     self.generate_rating_vector()
    #     recommendation = self.generate_similarity()
    #     return recommendation
    
    def reset_n_output(self, new_n_output):
        self.n_output = new_n_output
        return self.n_output
    