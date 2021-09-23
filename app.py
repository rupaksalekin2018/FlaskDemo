import os
import math
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory
app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

cnn_model_corn = tf.keras.models.load_model('static/testCorn_disease_detector_final.h5')
cnn_model_potato = tf.keras.models.load_model('static/FinalPotato_disease_detector_final.h5')
cnn_model_tomato = tf.keras.models.load_model('static/tomato_disease_detector.h5')
cnn_model_rice = tf.keras.models.load_model('static/rice_disease_detector_new.h5')

IMAGE_SIZE = 256

# Preprocess an image
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image /= 255.0  # normalize to [0,1] range

    return image


# Read the image from path and preprocess
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)

    return preprocess_image(image)


# Predict & classify image
def classify_corn(model, image_path):

    preprocessed_imgage = load_and_preprocess_image(image_path)
    preprocessed_imgage = tf.reshape(
        preprocessed_imgage, (1, IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    prob = cnn_model_corn.predict(preprocessed_imgage)

    return {"prob_1":prob, "class_name":"corn"}

def classify_potato(model, image_path):

    preprocessed_imgage = load_and_preprocess_image(image_path)
    preprocessed_imgage = tf.reshape(
        preprocessed_imgage, (1, IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    prob = cnn_model_potato.predict(preprocessed_imgage)

    return {"prob_1":prob, "class_name":"potato"}

def classify_tomato(model, image_path):

    preprocessed_imgage = load_and_preprocess_image(image_path)
    preprocessed_imgage = tf.reshape(
        preprocessed_imgage, (1, IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    prob = cnn_model_tomato.predict(preprocessed_imgage)

    return {"prob_1":prob, "class_name":"tomato"}

def classify_rice(model, image_path):

    preprocessed_imgage = load_and_preprocess_image(image_path)
    preprocessed_imgage = tf.reshape(
        preprocessed_imgage, (1, IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    prob = cnn_model_rice.predict(preprocessed_imgage)

    return {"prob_1":prob, "class_name":"rice"}

@app.route('/')
def index():
    return render_template('index.html', name=index)

@app.route('/impact')
def impact():
    return render_template('impact.html', name=impact)

@app.route('/explore')
def explore():
    return render_template('explore.html', name=explore)

@app.route('/contact')
def contact():
    return render_template('contact.html', name=contact)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prob = ""
    if request.method == 'GET':
        return render_template('predict.html', name=predict)
    else:
        option = request.form.getlist('group1')
        #print(option)
        selected_option = option[0]
        #print(selected_option)
        file = request.files["image"]
        if file.filename == "":
            error = 'Please Enter An Image!'
            return render_template('predict.html', error=error)

        kind = file.filename.split(".")
        ext = kind[1]
        valid_ext = ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']
        validate = ext in valid_ext
        if (validate == False):
            error = 'Please Enter An Valid Image Extension!'
            return render_template('predict.html', error=error)
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(upload_image_path)
        if (selected_option == "corn" ):
            prob = classify_corn(cnn_model_corn, upload_image_path)
            prob_values = prob['prob_1']
            prob_values_zero = list(prob_values[0])
            max_value = max(prob_values_zero)
            final_max_value = float("{:.2f}".format(max_value))
            max_index = prob_values_zero.index(max_value)
            print(max_index)
            class_name = prob['class_name']
            disease_names = ["Corn Gray Leaf Spot" , "Common Rust" , "Northern Leaf Blight" , "Healthy"]
            #final_values = {"Cercospora_leaf_spot Gray_leaf_spot":prob_values[0][0], "Common_rust": prob_values[0][1], "Northern_Leaf_Blight": prob_values[0][2], "Healthy": prob_values[0][3] }
            disease_index_name = disease_names[max_index]
            final_values = {"disease_name": disease_index_name , "value" : float(final_max_value) * 100}

        elif (selected_option == "potato"):
            prob = classify_potato(cnn_model_potato, upload_image_path)
            prob_values = prob['prob_1']
            prob_values_zero = list(prob_values[0])
            max_value = max(prob_values_zero)
            final_max_value = float("{:.2f}".format(max_value))
            max_index = prob_values_zero.index(max_value)
            print(max_index)
            class_name = prob['class_name']
            disease_names = ["Potato Early blight" , "Potato Late blight" , "Potato healthy" ]
            #final_values = {"Potato_Early_blight": prob_values[0][0] , "Potato_Late_blight": prob_values[0][1] , "Potato_healthy" : prob_values[0][2] }
            disease_index_name = disease_names[max_index]
            final_values = {"disease_name": disease_index_name , "value" : float(final_max_value) * 100}

        elif (selected_option == "tomato"):
            prob = classify_tomato(cnn_model_tomato, upload_image_path)
            prob_values = prob['prob_1']
            prob_values_zero = list(prob_values[0])
            max_value = max(prob_values_zero)
            final_max_value = float("{:.2f}".format(max_value))
            max_index = prob_values_zero.index(max_value)
            print(max_index)
            class_name = prob['class_name']
            disease_names = ["Late Blight" ,  "Leaf Mold", "Septoria Leaf Spot", "Spider mites Two spotted spider mite" , "Target Spot" , "Yellow Leaf Curl Virus" , "Mosaic virus",  "Healthy"]
            # final_values = {"Early Blight": prob_values[0][0] , "Healthy": prob_values[0][1] , "Late Blight": prob_values[0][2] , "Leaf Mold" : prob_values[0][3] , "Septoria Leaf Spot" : prob_values[0][4] , "Spider mites Two spotted spider mite" : prob_values[0][5] , "Target Spot" : prob_values[0][6] , "Mosaic virus" : prob_values[0][7] , "Yellow Leaf Curl Virus" : prob_values[0][8]  }
            disease_index_name = disease_names[max_index]
            final_values = {"disease_name": disease_index_name , "value" : float(final_max_value) * 100}

        elif (selected_option == "rice"):
            prob = classify_rice(cnn_model_rice, upload_image_path)
            prob_values = prob['prob_1']
            prob_values_zero = list(prob_values[0])
            max_value = max(prob_values_zero)
            final_max_value = float("{:.2f}".format(max_value))
            max_index = prob_values_zero.index(max_value)
            print(max_index)
            class_name = prob['class_name']
            disease_names = ["Bacterial Leaf Blight" ,  "Brown Plant Hopper", "Brown Spot", "False Smut" , "Healthy" , "Hispa" , "Neck Blast", "Sheath Blight Rot" , "Stemborer"]
            # final_values = {"Early Blight": prob_values[0][0] , "Healthy": prob_values[0][1] , "Late Blight": prob_values[0][2] , "Leaf Mold" : prob_values[0][3] , "Septoria Leaf Spot" : prob_values[0][4] , "Spider mites Two spotted spider mite" : prob_values[0][5] , "Target Spot" : prob_values[0][6] , "Mosaic virus" : prob_values[0][7] , "Yellow Leaf Curl Virus" : prob_values[0][8]  }
            disease_index_name = disease_names[max_index]
            final_values = {"disease_name": disease_index_name , "value" : float(final_max_value) * 100}
        # list1 = prob.tolist()
        # return render_template('predict.html', name=predict,list1=list1)
        return render_template('predict.html', name=predict, finals=final_values)
if __name__ == "__main__":
    app.run(debug = True)