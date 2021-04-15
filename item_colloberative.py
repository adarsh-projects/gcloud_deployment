import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import lite
from sklearn.model_selection import train_test_split

class RecomdModel:
	
	def __init__(self):
		self.ratingSource = 'https://raw.githubusercontent.com/adarsh-projects/GCP/main/appengine/docs/flexible/python/users_ratings.csv'
		self.productSource = 'https://raw.githubusercontent.com/adarsh-projects/GCP/main/appengine/docs/flexible/python/product.csv'
		
		#reading rating file
		self.rating_df = pd.read_csv(self.ratingSource)
		
		#reading product file
		self.product_df = pd.read_csv(self.productSource, encoding='unicode_escape')

		#merging both file
		self.ratings = pd.merge(self.rating_df, self.product_df, on='product_id')
		
		#After merging it gives result as String even if it is Integer
		#converting String to Integer
		self.ratings['product_id'] = self.ratings['product_id'].astype(str).astype(int)
		self.ratings['user_id'] = self.ratings['user_id'].astype(str).astype(int)
		self.ratings['rating'] = self.ratings['rating'].astype(str).astype(int)
		self.ratings['sex'] = self.ratings['sex'].astype(str).astype(int)
		self.ratings['brand'] = self.ratings['brand'].astype(str).astype(int)
		self.ratings.drop(['product_name','price'], inplace=True, axis=1)
		
		self.Xtrain, self.Xtest = train_test_split(self.ratings, test_size=0.2, random_state=1)
		self.Xtrain, self.Xvalidation = train_test_split(self.Xtrain, test_size=0.2, random_state=1)
		

		self.nproduct_id = self.rating_df.product_id.nunique()
		self.nuser_id = self.rating_df.user_id.nunique()
		self.embedding_dimension = 32
		
		#product input
		self.input_products = keras.layers.Input(shape=[1])
		self.embed_products = keras.layers.Embedding(self.nproduct_id+ 1, self.embedding_dimension)(self.input_products)
		self.products_out = keras.layers.Flatten()(self.embed_products)
		
		#user input
		self.input_users = keras.layers.Input(shape=[1])
		self.embed_user = keras.layers.Embedding(self.nuser_id + 1, self.embedding_dimension)(self.input_users)
		self.users_out = keras.layers.Flatten()(self.embed_user)
		
		#Sex_input
		self.input_sexs = keras.layers.Input(shape=[1])
		self.embed_sexs = keras.layers.Embedding(self.nuser_id + 1, self.embedding_dimension)(self.input_sexs)
		self.sexs_out = keras.layers.Flatten()(self.embed_sexs)
		
		#user rating
		self.input_ratings = keras.layers.Input(shape=[1])
		self.embed_ratings = keras.layers.Embedding(self.nuser_id+1, self.embedding_dimension)(self.input_ratings)
		self.ratings_out = keras.layers.Flatten()(self.embed_ratings)
		
		self.conc_layer = keras.layers.Concatenate()([self.products_out, self.users_out, self.ratings_out, self.sexs_out])
		self.x1 = keras.layers.Dense(128, activation='relu')(self.conc_layer)
		self.x1 = keras.layers.Dense( 64, activation='relu')(self.x1)
		self.x_out = self.x = keras.layers.Dense(2, activation='relu')(self.x1)
		
		self.model = keras.Model([self.input_products, self.input_users, self.input_ratings, self.input_sexs], self.x_out)
		opt = keras.optimizers.Adam(learning_rate=0.01)
		
		self.model.compile(optimizer=opt, loss='mean_absolute_error')
		self.model.summary()
		hist = self.model.fit( [self.Xtrain.product_id, self.Xtrain.user_id, self.Xtrain.rating, self.Xtrain.sex], self.Xtrain.brand, batch_size=1000, #change and try yourself
				epochs=5, #change and try ourself and find good model
				verbose=1, #0-silent 1-progress bar 2-one line per epoch
				validation_data=( [self.Xvalidation.product_id, self.Xvalidation.user_id, self.Xvalidation.rating, self.Xvalidation.sex], self.Xvalidation.brand ) )
				
		self.train_loss = hist.history['loss']
		self.val_loss = hist.history['val_loss']
		
		#self.model.save("efficuient_model")
		"""
		plt.plot(train_loss, color='r', label='Train Loss')
		plt.plot(val_loss, color='b', label='Validation Loss')
		plt.title("Train and  Validation Loss Curve")
		plt.legend()
		plt.show()
		"""
	def predictModel(self, user_id, rating, sex):
		#it collect all unique id of product in ratings csv file
		p_id = list(self.ratings.product_id.unique())
		product_arr = np.array(p_id) #get all book IDs
		
		#making recommendations for user 100
		# 100 is user id
		user = np.array([user_id for i in range(len(p_id))])
		ratings = np.array([rating for i in range(len(p_id))])
		sex = np.array([sex for i in range(len(p_id))])
		abcd = self.model.evaluate([self.Xtest.product_id, self.Xtest.user_id, self.Xtest.rating, self.Xtest.sex], self.Xtest.brand, batch_size=50, verbose=0)
		
		prediction = self.model.predict([product_arr, user, ratings, sex])
		prediction = prediction.reshape(-1) #reshape to single dimension
		#print(prediction)
		pred_ids = (-prediction).argsort()[0:5]
		
		#print(self.product_df['brand'].iloc[pred_ids])
		#brand_name = pd.read_csv('/home/adarsh/Desktop/recommend/colloberative/brand.csv')
		return self.product_df['brand'].iloc[pred_ids], abcd


model = RecomdModel()
"""
value = model.predictModel(2000, 3, 327)
print(value)
value = {"item": [x for x in value[0]]}
print(value)
	# 327 - male
	# 595 = female


entrypoint: gunicorn -b :$PORT main:app

runtime_config:
  python_version: 3


ERROR: (gcloud.app.deploy) Error Response: [8] Flex operation projects/demorecomd/regions/asia-south1/operations/2c5c2525-8208-4430-b59e-ae8a664b0cd5 error [RESOURCE_EXHAUSTED]: An internal error occurred while processing task /app-engine-flex/insert_flex_deployment/flex_create_resources>2021-04-11T15:37:56.712Z7845.fw.0: The requested amount of instances has exceeded GCE's default quota. Please see https://cloud.google.com/compute/quotas for more information on GCE resources






"""
