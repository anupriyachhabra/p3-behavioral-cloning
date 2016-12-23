from keras.utils.visualize_util import plot
from keras.models import model_from_json


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.compile("adam", "mse")
# got in second epoch for this model

weights_file = args.model.replace('json', 'h5')
model.load_weights(weights_file)

plot(model, to_file='model.png', show_shapes=True)