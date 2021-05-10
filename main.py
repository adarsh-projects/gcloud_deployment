from flask import Flask, request, jsonify, render_template
from item_colloberative import RecomdModel
import pandas as pd
import json

model = RecomdModel()

app = Flask("__name__")

@app.errorhandler(404)
def not_found(error):
    return "<h2>404</h2><p>Nahi Mila</p>"

@app.route('/')
def apitest():
	return render_template('index.html', query="")

@app.route('/predict', methods=['POST'])
def predict():
	#query = [ x for x in request.form.values()]
	
	# user rating sex
	#value = model.predictModel(int(query[0]), int(query[1]), int(query[2]))
	value = model.predictModel(180, 3, 327)
	
	value = {"brand": [x for x in value[0]]}
	return value #{"item": "hey Adarsh"}
	#return {"item": "Hey, Adarsh"}

if __name__ == '__main__':
	app.run()
