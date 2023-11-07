from keras.models import load_model

def get_model():

    # Load the model
    model = load_model("keras_nonresized/keras_model.h5", compile=False)

    # Load the labels
    class_names = open("keras_nonresized/labels.txt", "r").readlines()

    return model, class_names