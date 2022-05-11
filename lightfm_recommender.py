import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from scipy.sparse import coo_matrix
from annoy import AnnoyIndex
from Split import train_test_split



ratings_df = pd.read_csv("ratings.csv")
movies_df = pd.read_csv("movies.csv")




def get_movie_info_by_id(movie_id):
    for i in range(movies_df.shape[0]):
        if(movies_df.loc[i,"movieId"] == movie_id):
            title = movies_df.loc[i,"title"]
            genre = movies_df.loc[i,"genres"]      
            return (title,genre)

def nearest_movies_annoy(movie_id, index, n=10, print_output=True):
    nn = index.get_nns_by_item(movie_id, 10)
#     print(nn)
    if print_output:
        for movie_id in nn:
            print("\n", get_movie_info_by_id(movie_id))

def sample_recommendation(user_ids, model, data_training , n_items=10, print_output=True):
    n_users, n_items = data_training.shape
    
    for user_id in user_ids:
        known_positives = [data_training.tocsr()[user_id].indices]
        top_items_ids = [i for i in annoy_member_idx.get_nns_by_vector(np.append(user_embeddings[user_id], 0), 10)]
    return top_items_ids


######################  Preparing Dataset ################################
dataset = Dataset()
dataset.fit(ratings_df["userId"],
            ratings_df["movieId"])


num_users, num_items = dataset.interactions_shape()
print(f'Num users: {num_users}, num_items {num_items}.')


feedbacks_list = []
for i in range(0,ratings_df.shape[0]):
    feedbacks_list.append((ratings_df.loc[i,"userId"] , ratings_df.loc[i,"movieId"] , ratings_df.loc[i,"rating"]))
(interactions, weights) = dataset.build_interactions(feedbacks_list)



Row = weights.row
Col = weights.col
Value = weights.data

train, test = train_test_split(Row, Col, Value, train_size=0.75, random_state=None)
train = coo_matrix(train)
test = coo_matrix(test) # full interaction matrix
test = test - train


######################  Model Building and Training  ###################################


from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score

model = LightFM(learning_rate = 0.05, loss='warp', no_components=10, item_alpha=0.001)
model.fit(train, epochs=10)

# print(train.toarray())
train_precision = precision_at_k(model, train, k=5).mean()
train_auc = auc_score(model, train).mean()

# print(test.toarray())
test_precision = precision_at_k(model, test, k=5).mean()
test_auc = auc_score(model, test).mean()

print("train precision:" , round(train_precision,2) , "****  test precision:" , round(test_precision,2))
print("train AUC:" , round(train_auc,2), "****  test AUC:", round(test_auc,2))



##############################  Recommendation  ###########################
_, item_embeddings = model.get_item_representations()
print(item_embeddings.shape)



factors = item_embeddings.shape[1] # Length of item vector that will be indexed
annoy_idx = AnnoyIndex(factors)  
for i in range(item_embeddings.shape[0]):
    v = item_embeddings[i]
    annoy_idx.add_item(i, v)

annoy_idx.build(10) # 10 trees
annoy_idx.save('movielens_item_Annoy_idx4.ann')



# Basically we add a nomalizing 
# factor to each item vector - making their distances equal with each other. Then when we query with a
# user vector, we add a 0 to the end, and the result is proportional to the inner producct of the user 
# and item vectors. This is a sneaky way to do an aproximate maximum inner product search.

norms = np.linalg.norm(item_embeddings, axis=1)
max_norm = norms.max()
extra_dimension = np.sqrt(max_norm ** 2 - norms ** 2)
norm_data = np.append(item_embeddings, extra_dimension.reshape(norms.shape[0], 1), axis=1)

#First an Annoy index:

user_factors = norm_data.shape[1]
annoy_member_idx = AnnoyIndex(user_factors)  # Length of item vector that will be indexed

for i in range(norm_data.shape[0]):
    v = norm_data[i]
    annoy_member_idx.add_item(i, v)
    
annoy_member_idx.build(10)



_ , user_embeddings = model.get_user_representations()


recommendation_movie_ids = sample_recommendation([3,4], model, train, print_output=True)
for movie_id in recommendation_movie_ids:
   print(get_movie_info_by_id(movie_id))