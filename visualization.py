from keras.utils.visualize_util import plot
from keras.models import model_from_json


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

plot(model, to_file='model.png', show_shapes=True)
