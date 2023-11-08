from keras.models import load_model

def get_model():

    # Load the model
    model = load_model("/home/raspberry/Documents/clip_classification/Clip_CNN/keras_model.h5", compile=False)

    # Load the labels
    class_names = open("/home/raspberry/Documents/clip_classification/Clip_CNN/labels.txt", "r").readlines()

    return model, class_names