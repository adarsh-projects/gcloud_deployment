import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import lite
from sklearn.model_selection import train_test_split

ratingSource = 'https://raw.githubusercontent.com/adarsh-projects/gcloud_deployment/main/users_ratings.csv'
productSource = 'https://raw.githubusercontent.com/adarsh-projects/gcloud_deployment/main/product.csv'

#reading rating file
rating_df = pd.read_csv(ratingSource)

#reading product file
product_df = pd.read_csv(productSource, encoding='unicode_escape')

#merging both file
ratings = pd.merge(rating_df, product_df, on='product_id')

#After merging it gives result as String even if it is Integer
#converting String to Integer
ratings['product_id'] = ratings['product_id'].astype(str).astype(int)
ratings['user_id'] = ratings['user_id'].astype(str).astype(int)
ratings['rating'] = ratings['rating'].astype(str).astype(int)
ratings['sex'] = ratings['sex'].astype(str).astype(int)
ratings['brand'] = ratings['brand'].astype(str).astype(int)
ratings.drop(['product_name','price'], inplace=True, axis=1)

Xtrain, Xtest = train_test_split(ratings, test_size=0.2, random_state=1)
Xtrain, Xvalidation = train_test_split(Xtrain, test_size=0.2, random_state=1)

nproduct_id = rating_df.product_id.nunique()
nuser_id = rating_df.user_id.nunique()
embedding_dimension = 32

#product input
input_products = keras.layers.Input(shape=[1])
embed_products = keras.layers.Embedding(nproduct_id+ 1, embedding_dimension)(input_products)
products_out = keras.layers.Flatten()(embed_products)

#user input
input_users = keras.layers.Input(shape=[1])
embed_user = keras.layers.Embedding(nuser_id + 1, embedding_dimension)(input_users)
users_out = keras.layers.Flatten()(embed_user)

#Sex_input
input_sexs = keras.layers.Input(shape=[1])
embed_sexs = keras.layers.Embedding(nuser_id + 1, embedding_dimension)(input_sexs)
sexs_out = keras.layers.Flatten()(embed_sexs)

#user rating
input_ratings = keras.layers.Input(shape=[1])
embed_ratings = keras.layers.Embedding(nuser_id+1, embedding_dimension)(input_ratings)
ratings_out = keras.layers.Flatten()(embed_ratings)

conc_layer = keras.layers.Concatenate()([products_out, users_out, ratings_out, sexs_out])
x1 = keras.layers.Dense(128, activation='relu')(conc_layer)
x1 = keras.layers.Dense( 64, activation='relu')(x1)
x_out = x = keras.layers.Dense(2, activation='relu')(x1)

model = keras.Model([input_products, input_users, input_ratings, input_sexs], x_out)
opt = keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer=opt, loss='mean_absolute_error')
model.summary()
hist = model.fit( [Xtrain.product_id, Xtrain.user_id, Xtrain.rating, Xtrain.sex], Xtrain.brand, batch_size=1000, #change and try yourself
		epochs=5, #change and try ourself and find good model
		verbose=1, #0-silent 1-progress bar 2-one line per epoch
		validation_data=( [Xvalidation.product_id, Xvalidation.user_id, Xvalidation.rating, Xvalidation.sex], Xvalidation.brand ) )

#model.save("//home//adarsh//Desktop//Flask//abcd");
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']

p_id = list(ratings.product_id.unique())

product_arr = np.array(p_id) #get all book IDs

user = np.array([1000 for i in range(len(p_id))])
ratings = np.array([5 for i in range(len(p_id))])
sex = np.array([327 for i in range(len(p_id))])

abcd = model.evaluate([Xtest.product_id, Xtest.user_id, Xtest.rating, Xtest.sex], Xtest.brand, batch_size=50, verbose=0)

prediction = model.predict([product_arr, user, ratings, sex])
prediction = prediction.reshape(-1) #reshape to single dimension

print(prediction)
pred_ids = (-prediction).argsort()[0:5]

print("pred: ", pred_ids)
print("prod: ",product_df['brand'].iloc[pred_ids])
#value = predictModel(2000, 3, 327)

