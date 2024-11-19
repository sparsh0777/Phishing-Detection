from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import pandas as pd
from file_operations import file_methods
from extract_features.get_features_df import get_features
from app_logging.logger import create_log
import pickle
from data_preprocessing import preprocessing
import numpy as np

import __main__
__main__.file_methods = file_methods
__main__.get_features = get_features
__main__.preprocessing = preprocessing
__main__.create_log = create_log

upload_folder = "App_Uploads/"
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = upload_folder


# Specifying GET method
@app.route('/', methods=['GET'])
def home():
    return render_template('new.html')

@app.route('/detect-site', methods=['GET'])
def detectPhishing():
    return render_template('DetectPhishing.html')

@app.route('/home', methods=['GET'])
def newHome():
    return render_template('new.html')

@app.route('/about', methods=['GET'])
def About():
    return render_template('About.html')

# Specifying POST method
@app.route("/predict", methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            url = str(request.form['website'])
            uploaded_file = request.files['file']
            
            if url != '':
                data = get_features(url)
            else:
                if len(os.listdir(upload_folder)) != 0:
                    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], os.listdir(upload_folder)[0]))
                filename = secure_filename(uploaded_file.filename)
                uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                file_extension = os.listdir(upload_folder)[0].split('.')[-1]
                raw_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if file_extension == 'csv':
                    raw_data = pd.read_csv(raw_file_path, header=None)
                else:
                    raw_data = pd.read_excel(raw_file_path, header=None, engine='openpyxl')
                raw_data.columns = ['URL']
                map_url = raw_data.to_dict()['URL']
                print(map_url)
                data = get_features(raw_data)
            
            # Loading categorical columns' list
            with open("EncoderPickle/cat_col_list.txt", 'rb') as file:
                cat_col_list = pickle.load(file)

            # Loading continuous columns' list
            with open("EncoderPickle/cont_col_list.txt", 'rb') as file:
                cont_col_list = pickle.load(file)

            # Getting encoded values for categorical data
            preprocessor = preprocessing.Preprocessor("App_Logs/output", create_log)
            data = preprocessor.encode_categorical_values_prediction(data, cat_col_list)

            print(data.columns)
            print(cont_col_list)
            # Replacing -1 with -999 in continuous columns
            data = preprocessor.replace_missing_values(data, cont_col_list)

            # Loading the kmeans' scaler
            with open("EncoderPickle/kmeans_scaler.txt", 'rb') as file:
                scaler = pickle.load(file)

            # Scaling continuous columns for kmeans algo
            scaled_cont_data = pd.DataFrame(scaler.transform(data[cont_col_list]))
            scaled_cont_data.columns = cont_col_list

            file_object = "App_Logs/output"
            log_function = create_log
            file_loader = file_methods.FileOperation(file_object, log_function)
            kmeans = file_loader.load_model('KMeans')
            clusters = kmeans.predict(scaled_cont_data)
            data['Clusters'] = clusters
            clusters = data['Clusters'].unique()
            
            # Loading columns/features that has been used for training
            with open("EncoderPickle/col_to_keep.txt", 'rb') as file:
                col_to_keep = pickle.load(file)

            # Loading a dictionary containing dropped columns and
            # standardization pipelines of each cluster derived during training
            with open('EncoderPickle/drop_standardization_pipeline.txt', 'rb') as file:
                drop_pipeline_dict = pickle.load(file)

            # adding missing columns from col_to_keep as some categorical features
            # may not contain all the possible values available in training set
            add_col_list = [col for col in col_to_keep if col not in data.columns]
            if len(add_col_list) != 0:
                for col in add_col_list:
                    data[col] = 0

            result = []
            index_list = []
            for i in clusters:
                cluster_data = data[data['Clusters'] == i]
                # cluster_data = cluster_data.drop(['Clusters'], axis=1)
                cluster_data = cluster_data[col_to_keep]
                if len(drop_pipeline_dict[i]['drop_cols']) != 0:
                    cluster_data = cluster_data.drop(columns=drop_pipeline_dict[i]['drop_cols'])
                cont_col_list = drop_pipeline_dict[i]['cont_col']
                for col in cont_col_list:
                    cluster_data[col] = cluster_data[col].apply(lambda x: np.log(x + 1000))
                cluster_data[cont_col_list] = drop_pipeline_dict[i]['scaler'].transform(cluster_data[cont_col_list])
                temp_index = list(cluster_data.index)
                index_list.extend(temp_index)
                model_name = file_loader.find_correct_model_file(i)
                model = file_loader.load_model(model_name)
                for val in model.predict(cluster_data):
                    result.append(val)
            
            if url != '':
                text_result = ['yes' if x == 1 else 'no' for x in result][0]
                decision_text = f"Is {url} a phishing website? Answer: {text_result}"
                return render_template('DetectPhishing.html', prediction_text=decision_text)
            else:
                result = pd.DataFrame(result, columns=['Phishing'])
                result['URL'] = index_list
                result['URL'] = result['URL'].map(map_url)
                result['Phishing'] = result['Phishing'].map({0: 'no', 1: 'yes'})
                result = result[['URL', 'Phishing']]
                result.to_csv("App_Output/output.csv", index=False)
                path = "App_Output/output.csv"
                return send_file(path, as_attachment=True)
        else:
            return render_template('DetectPhishing.html')
    except Exception as e:
        return render_template('DetectPhishing.html', prediction_text=str(e))


@app.route("/predict-api", methods=['POST'])
def predictApi():
    if request.method == 'POST':
        url = str(request.form['website'])
        print(url)
        #uploaded_file = request.files['file']
        
        if url != '':
            data = get_features(url)
        
        # Loading categorical columns' list
        with open("EncoderPickle/cat_col_list.txt", 'rb') as file:
            cat_col_list = pickle.load(file)

        # Loading continuous columns' list
        with open("EncoderPickle/cont_col_list.txt", 'rb') as file:
            cont_col_list = pickle.load(file)

        # Getting encoded values for categorical data
        preprocessor = preprocessing.Preprocessor("App_Logs/output", create_log)
        data = preprocessor.encode_categorical_values_prediction(data, cat_col_list)

        print(data.columns)
        print(cont_col_list)
        # Replacing -1 with -999 in continuous columns
        data = preprocessor.replace_missing_values(data, cont_col_list)

        # Loading the kmeans' scaler
        with open("EncoderPickle/kmeans_scaler.txt", 'rb') as file:
            scaler = pickle.load(file)

        # Scaling continuous columns for kmeans algo
        scaled_cont_data = pd.DataFrame(scaler.transform(data[cont_col_list]))
        scaled_cont_data.columns = cont_col_list

        file_object = "App_Logs/output"
        log_function = create_log
        file_loader = file_methods.FileOperation(file_object, log_function)
        kmeans = file_loader.load_model('KMeans')
        clusters = kmeans.predict(scaled_cont_data)
        data['Clusters'] = clusters
        clusters = data['Clusters'].unique()
        
        # Loading columns/features that has been used for training
        with open("EncoderPickle/col_to_keep.txt", 'rb') as file:
            col_to_keep = pickle.load(file)

        # Loading a dictionary containing dropped columns and
        # standardization pipelines of each cluster derived during training
        with open('EncoderPickle/drop_standardization_pipeline.txt', 'rb') as file:
            drop_pipeline_dict = pickle.load(file)

        # adding missing columns from col_to_keep as some categorical features
        # may not contain all the possible values available in training set
        add_col_list = [col for col in col_to_keep if col not in data.columns]
        if len(add_col_list) != 0:
            for col in add_col_list:
                data[col] = 0

        result = []
        index_list = []
        for i in clusters:
            cluster_data = data[data['Clusters'] == i]
            # cluster_data = cluster_data.drop(['Clusters'], axis=1)
            cluster_data = cluster_data[col_to_keep]
            if len(drop_pipeline_dict[i]['drop_cols']) != 0:
                cluster_data = cluster_data.drop(columns=drop_pipeline_dict[i]['drop_cols'])
            cont_col_list = drop_pipeline_dict[i]['cont_col']
            for col in cont_col_list:
                cluster_data[col] = cluster_data[col].apply(lambda x: np.log(x + 1000))
            cluster_data[cont_col_list] = drop_pipeline_dict[i]['scaler'].transform(cluster_data[cont_col_list])
            temp_index = list(cluster_data.index)
            index_list.extend(temp_index)
            model_name = file_loader.find_correct_model_file(i)
            model = file_loader.load_model(model_name)
            for val in model.predict(cluster_data):
                result.append(val)
        
        if url != '':
            text_result = ['yes' if x == 1 else 'no' for x in result][0]
            decision_text = f"Is {url} a phishing website? Answer: {text_result}"
            return jsonify({"output" : decision_text})
    else:
        return render_template('index.html')

# port = int(os.getenv('PORT'))
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=3004)
