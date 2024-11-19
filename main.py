from wsgiref import simple_server
from flask import Flask, request
from flask import Response
import os
from flask_cors import CORS, cross_origin
from prediction_validation_insertion import PredictionValidation
from training_model import TrainModel
from training_validation_insertion import TrainValidation
import flask_monitoringdashboard as dashboard
from predict_from_model import Prediction

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
dashboard.bind(app)
CORS(app)


@app.route("/predict", methods=['POST'])
@cross_origin()
def predict_route_client():
    try:
        if request.json['folderPath'] is not None:
            path = request.json['folderPath']
            # Object initialization
            prediction_val = PredictionValidation(path)
            # Calling the prediction_validation function
            prediction_val.prediction_validation()
            # Object initialization
            prediction = Prediction(path)
            # Predicting for dataset present in database
            path = prediction.prediction_from_model()
            return Response(f"Prediction file created at {path}")
    except ValueError:
        return Response(f"Error occurred: {ValueError}")
    except KeyError:
        return Response(f"Error occurred: {KeyError}")
    except Exception as e:
        return Response(f"Error occurred: {e}")


@app.route("/train", methods=['POST'])
@cross_origin()
def train_route_client():
    try:
        if request.json['folderPath'] is not None:
            path = request.json['folderPath']
            # Object initialization
            train_val_obj = TrainValidation(path)
            # Calling the training_validation function
            train_val_obj.train_validation()
            # Object initialization
            train_model_obj = TrainModel()
            # Training the model for the files in the table
            train_model_obj.training_model()
    except ValueError:
        return Response(f"Error Occurred: {ValueError}")
    except KeyError:
        return Response(f"Error Occurred: {KeyError}")
    except Exception as e:
        return Response(f"Error Occurred: {e}")
    return Response("Training successfull!!")

#port = int(os.getenv("PORT"))
if __name__ == "__main__":
    host = '0.0.0.0'
    port = 5000
    httpd = simple_server.make_server(host, port, app)
    print("Serving on %s %d" % (host, port))
    httpd.serve_forever()
