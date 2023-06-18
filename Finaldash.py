import os
import glob
import math
import pickle
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.stats import randint
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import cv2
import git
import subprocess
from io import BytesIO
import base64
from flask import Flask, request

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import dash_table
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Output, Input
from dash_canvas import DashCanvas, utils as canvas_utils

import plotly.graph_objects as go

from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from dash_canvas.utils import parse_jsonstring


# Functions to handle dataset
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

def make_dataset(size=28):
    repo_url = "https://github.com/pongsapaks/Thai-handwrittingnumberproject.git"
    repo_dir = "Thai-handwrittingnumberproject"
    subprocess.run(["git", "clone", repo_url, repo_dir])

    image_dir = os.path.join(repo_dir, "raw2")
    image_files = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith(".png"):
                image_files.append(os.path.join(root, file))

    print("Total image files:", len(image_files))

    X = []
    Y = []

    for image_path in image_files:
        img = cv2.imread(image_path)
        img = Image.open(image_path).convert("L")
        img = ImageOps.invert(img)
        img = img.resize((size, size))
        label = os.path.basename(os.path.dirname(image_path))
        x = np.array(img)
        X.append(x)
        Y.append(label)

    X = np.asarray(X)
    Y = np.asarray(Y)
    data = {"X": X, "Y": Y}
    pickle.dump(data, open("thainumber_{}.pkl".format(size), "wb"), protocol=2)

def make_dataset2(size=28):
    repo_url = "https://github.com/pongsapaks/Thai-handwrittingnumberproject.git"
    repo_dir = "Thai-handwrittingnumberproject"
    subprocess.run(["git", "clone", repo_url, repo_dir])

    image_dir = os.path.join(repo_dir, "raw2")
    image_files = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith(".png"):
                image_files.append(os.path.join(root, file))

    print("Total image files:", len(image_files))

    X = []
    Y = []

    for image_path in image_files:
        img = cv2.imread(image_path)
        img = Image.open(image_path).convert("L")
        img = ImageOps.invert(img)
        img = img.resize((size, size))
        label = os.path.basename(os.path.dirname(image_path))
        x = np.array(img)
        X.append(x)
        Y.append(label)

    X = np.asarray(X)
    Y = np.asarray(Y)

    reshaped_X = X.reshape((X.shape[0], -1))
    Ydf = pd.DataFrame(Y)
    Xdf = pd.DataFrame(reshaped_X)

    X_mean = Xdf.mean()
    X_std = Xdf.std()
    Z = (Xdf - X_mean) / X_std
    Z = Z.fillna(0)

    with open("std_params.pkl", "wb") as f:
        pickle.dump((X_mean, X_std), f)

    pca = PCA(n_components=0.75)
    pca.fit(Z)
    X_pca = pca.transform(Z)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, Y, test_size=0.2, random_state=42)

    sv = SVC(C=10, gamma=0.001, kernel='rbf')
    sv.fit(X_train, y_train)
    pred = sv.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print(f"Model accuracy: {accuracy}")

    with open("model.pkl", "wb") as f:
        pickle.dump(sv, f)
    with open("pca.pkl", "wb") as f:
        pickle.dump(pca, f)
make_dataset2()

def load_dataset(size=28):
    data = pickle.load(open("thainumber_{}.pkl".format(size), "rb"))
    X = data['X']
    Y = data['Y']
    return X, Y


def prepare_input(file):
    img = load_img(file, grayscale=True, target_size=(28, 28))
    img = ImageOps.invert(img)
    x = img_to_array(img)
    return x


def img_cloud_dataset(size=28):
    X, Y = load_dataset(size)
    x = 0
    y = 0
    new_im = Image.new('L', (size * 50, size * math.ceil(X.shape[0] / 50)))
    for i in range(0, X.shape[0]):
        if (i != 0 and i % 50 == 0):
            y += size
            x = 0

        im = array_to_img(X[i])
        new_im.paste(im, (x, y))
        x += size
    new_im.save("cloud_dataset_{}.png".format(size))

def preprocess_image(image_path, size=28):
    img = Image.open(image_path).convert("L")
    img = ImageOps.invert(img)
    img = img.resize((size, size))
    img_arr = np.array(img)
    return img_arr

def predict_image(image, model_path="model.pkl", pca_path="pca.pkl", std_params_path="std_params.pkl", size=28):
    # Load the model, pca, and standardization parameters from files
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)
    with open(std_params_path, 'rb') as f:
        X_mean, X_std = pickle.load(f)

    # Preprocess the image
    # If image is not a path but already an image, skip this line
    if isinstance(image, str):
        image = Image.open(image).convert("L")
    image = ImageOps.invert(image)
    image = image.resize((size, size))
    img_arr = np.array(image)

    # Flatten and standardize the image
    reshaped_image = img_arr.reshape((1, -1))
    # Avoid division by zero by adding a small constant to X_std
    epsilon = 1e-8
    standardized_image = (reshaped_image - np.array(X_mean).reshape(1,-1)) / (np.array(X_std).reshape(1,-1) + epsilon)

    if np.isnan(standardized_image).any():
        print("standardized_image still contains NaN values!")

    # Apply PCA
    transformed_image = pca.transform(standardized_image)

    # Predict the label
    prediction = model.predict(transformed_image)

    return prediction

make_dataset()

X, Y = load_dataset()

# Data preprocessing
reshaped_X = X.reshape((X.shape[0], -1))
Ydf = pd.DataFrame(Y)
Xdf = pd.DataFrame(reshaped_X)

X_mean = Xdf.mean()
X_std = Xdf.std()
Z = (Xdf-X_mean)/X_std
Z = Z.fillna(0)

pca = PCA(n_components = 0.75)
pca.fit(Z)
X_pca = pca.transform(Z)

X_train, X_test, y_train, y_test = train_test_split(X_pca, Y, train_size=0.8, random_state=42)

# SVM model
sv_fix = SVC(C=10, gamma=0.001, kernel='rbf', probability=True)
sv_fix.fit(X_train, y_train)

pred = sv_fix.predict(X_test)
accuracy = accuracy_score(y_test, pred)

report = classification_report(y_test, pred,digits=3)
report_data = []
lines = report.split('\n')
for line in lines[2:-3]:
    row_data = line.split()
    report_data.append(row_data)
df_report = pd.DataFrame(report_data, columns=['class', 'precision', 'recall', 'f1-score', 'support'])

label_binarizer = LabelBinarizer().fit(y_train)
y_onehot_test = label_binarizer.transform(y_test)
y_score = sv_fix.predict_proba(X_test)

n_classes = 0
target_names = np.unique(y_train)
target_names = np.sort(np.where(target_names == '10', '0', target_names))

# Confusion matrix
cm = confusion_matrix(y_test, pred)
labels = np.unique(y_test)

trace = go.Heatmap(x=labels, y=labels, z=cm, colorscale="Blues", colorbar=dict(title="Count"), xgap=1, ygap=1, text=cm, hoverinfo="text")
layout = go.Layout(title="Confusion Matrix", xaxis=dict(title="Predicted labels"), yaxis=dict(title="True labels"), template="plotly_white",)

# Calculate the false positive rate (fpr) and true positive rate (tpr) for the ROC curve
fpr, tpr, _ = roc_curve(y_onehot_test[:, n_classes], y_score[:, n_classes])
roc_auc = auc(fpr, tpr)


# Create the number radio items
number_radio_items = dcc.RadioItems(
    id='number-radio-items',
    options=[{'label': str(num), 'value': num} for num in range(10)],
    value=n_classes,
    inline=True
)

# Define the slider
train_test_split_slider = dcc.Slider(
    id='train-test-slider',
    min=0,
    max=100,
    step=10,
    value=80,
    marks={i: f'{i}' for i in range(0, 101, 10)},
)
canvas = html.Div([
        html.Div('Canvas', className="text-center"),
        DashCanvas(
            id='canvas',
            lineWidth=5,
            width=300,
            height=300,
            hide_buttons=['rectangle', 'line', 'zoom', 'pan', 'select'],
            lineColor='black',
        ),
])

sidebar = dbc.Card(
    [
        dbc.CardHeader(html.Div("Input Model", className="text-center")),
        dbc.CardBody(
            [
                html.Div(canvas, style={'margin': '10px'}),
                daq.StopButton('Predict', id='button_predict', n_clicks=0),
                html.Div(id='prediction'),
                daq.LEDDisplay(
                    label="Dataset",
                    value=X_pca.shape[0],  # Pass the integer value directly
                    style={'font-size': '12px'}
                ),
                daq.LEDDisplay(
                    id='train-data-led',
                    label="Train data",
                    value=X_train.shape[0],
                    style={'font-size': '12px'}
                ),
                daq.LEDDisplay(
                    id='test-data-led',
                    label="Test data",
                    value=X_test.shape[0],
                    style={'font-size': '12px'}
                ),
                daq.LEDDisplay(
                    label="Class",
                    value=str(np.unique(Y).shape[0]),
                    style={'font-size': '12px'}
                ),
                html.Div("Select Number to change ROC Curve", style={'margin': '10px'}),
                html.Div(number_radio_items, style={'margin': '10px'}),
                html.Label("Select your % of training data"),
                html.Div("Train-Test Split:", style={'margin': '10px'}),
                html.Div(train_test_split_slider, style={'margin': '10px'}),
                html.Div(id='slider-output-container'),
            ]
        ),
    ],
    style={'margin': '10px'},
)

header = dbc.Card(
    [
        dbc.CardHeader(html.H1("Thai handwritten number Recognition", className="text-center"))
    ]
)

# Define the main content layout
content = dbc.Card(
    [
        dbc.CardHeader(html.Div("ROC Curve", className="card-text")),
        dbc.CardBody(
            [
                dcc.Graph(
                    id='roc-curve',
                    config={'displayModeBar': False},
                    style={'width':'100%'}, # Adjust based on your preference
                    figure={
                        'layout': {
                            'xaxis': {'scaleanchor': 'y'}, # This ensures a 1:1 aspect ratio
                            'yaxis': {'scaleanchor': 'x'}, # This ensures a 1:1 aspect ratio
                            'autosize': True, # This should let the plot fill the space
                            'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50}, # You may need to adjust these values
                        }
                    }
                )
            ]
        )
    ]
)

content2 = dbc.Card([
    dbc.CardBody([
        dbc.Card([
            dbc.CardHeader("Confusion Matrix"),
            dbc.CardBody(dcc.Graph(id='confusion-matrix', config={'displayModeBar': False}))
        ]),
    ])
])
content3 = dbc.Card([
                dbc.CardHeader("Classification Report"),
                dbc.CardBody([
                    html.Div(
                        [
                            # Accuracy graph remaining callback when train-test changes
                            daq.GraduatedBar(
                                id='model-accuracy',
                                label="Model Accuracy",
                                max=100,
                                color={"ranges": {"green": [70, 100], "yellow": [50, 70], "red": [0, 50]}},
                                showCurrentValue=True,
                                value= accuracy*100,
                            ),
                        ],
                        style={'width': '100%'} 
                    ),
                    dash_table.DataTable(id='report-table')
                ])
            ]),


# Initialize the Flask app
server = Flask(__name__)

@server.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    image_data = data['image']
    image_data = base64.b64decode(image_data.split(',')[1])
    image = Image.open(BytesIO(image_data))

    # preprocess and predict
    prediction = predict_image(image) # Modify this line to suit the preprocessing and prediction in your case
    return str(prediction)

# Dash app initialization
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(header,width=12)]),
    dbc.Row([
        dbc.Col(sidebar, width=3),
        dbc.Col([
            dbc.Row([
                dbc.Col(content2, width=12)
                ]),
            dbc.Row([
                dbc.Col(content, width=6),
                dbc.Col(content3, width=6)
            ])])
    ], align="start"),
], fluid=True)


@app.callback(
    dash.dependencies.Output('prediction', 'children'),
    [dash.dependencies.Input('button_predict', 'n_clicks')],
    [dash.dependencies.State('canvas', 'json_data')]
)
def update_output(n_clicks, json_data):
    print("Button clicked")  # This line will print when the button is clicked
    if n_clicks > 0 and json_data is not None:
        print("Data received")  # This line will print if json_data is not None

        image_path = f"drawn_image_{n_clicks}.png"
        mask = parse_jsonstring(json_data)
        image = Image.fromarray(mask.astype('uint8') * 255)
        image.save(image_path)
        print(f"Image saved as {image_path}")  # This line will print the image save info

        prediction = predict_image(image)
        print("Prediction made:", prediction)  # This line will print the prediction
        return html.Div([
            'Predicted number: {}'.format(prediction),
            html.Img(src=image_path)  # Display the image that was used for prediction
        ])
    else:
        print("No data received")  # This line will print if json_data is None
        return 'Please draw on the canvas before predicting'
    
@app.callback(
    dash.dependencies.Output('roc-curve', 'figure'),
    dash.dependencies.Input('number-radio-items', 'value')
)
def update_roc_curve(selected_number):
    # Calculate the false positive rate (fpr) and true positive rate (tpr) for the ROC curve
    fpr, tpr, _ = roc_curve(y_onehot_test[:, selected_number], y_score[:, selected_number])
    roc_auc = auc(fpr, tpr)

    # Update the ROC curve trace
    roc_trace = go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f"ROC curve for {target_names[selected_number]} (AUC = {roc_auc:.2f})",
        line=dict(color='red')
    )

    # Update the layout for the ROC curve
    layout = go.Layout(
        title=f"{target_names[selected_number]} vs Rest multiclass",
        xaxis={'title': 'False Positive Rate'},
        yaxis={'title': 'True Positive Rate'},
        hovermode='closest',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return {'data': [roc_trace], 'layout': layout}




# Callback function to update the report

@app.callback(
    [
        Output('slider-output-container', 'children'),
        Output('model-accuracy', 'value'),
        Output('report-table', 'data'),
        Output('report-table', 'columns'),
        Output('confusion-matrix', 'figure')  
    ],
    [Input('train-test-slider', 'value')]
)

def update_output(slider_value):
    # Convert slider_value to proportion
    train_size = slider_value / 100.0
    
    # Update the train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_pca, Y, train_size=train_size, random_state=42)
    
    # Train the model
    sv = SVC(C=10, gamma=0.001, kernel='rbf')
    sv.fit(X_train, y_train)

    # Make predictions
    y_pred_sv = sv.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred_sv) * 100

    # Generate classification report
    report = classification_report(y_test, y_pred_sv, output_dict=True,digits=3)
    report_data = []
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            metrics['class'] = label
            report_data.append(metrics)
            df_report = pd.DataFrame(report_data)
    
    df_report = df_report.round(2)
    
    # Create columns for the DataTable
    columns = [{"name": col, "id": col} for col in df_report.columns]

    # Create data for the DataTable
    data = df_report.to_dict('records')

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred_sv)

    # Define the labels
    labels = np.unique(y_test)

    # Create a heatmap trace
    trace = go.Heatmap(
        x=labels,
        y=labels,
        z=cm,
        colorscale="Blues",
        colorbar=dict(title="Count"),
        xgap=1,
        ygap=1,
        text=cm,
        hoverinfo="text"
    )

    # Create the layout
    layout = go.Layout(
        xaxis=dict(title="Predicted labels"),
        yaxis=dict(title="True labels"),
        template="plotly_white",
    )

    # Create the figure
    fig = go.Figure(data=[trace], layout=layout)   
    return 'Train : Test = {} : {}'.format(slider_value, 100-slider_value), accuracy, data, columns, fig

def update_graphs(train_size):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_pca, Y, train_size=train_size, random_state=42)
    return accuracy*100, report_data, cm

def update_confusion_matrix(train_test_split_value):
    # Calculate the confusion matrix
    pred = sv_fix.predict(X_test)
    cm = confusion_matrix(y_test, pred)
    labels = np.unique(y_test)
    # Create the heatmap
    trace = go.Heatmap(x=labels, y=labels, z=cm, colorscale="Blues", colorbar=dict(title="Count"), xgap=1, ygap=1, text=cm, hoverinfo="text")
    layout = go.Layout(title="Confusion Matrix", xaxis=dict(title="Predicted labels"), yaxis=dict(title="True labels"), template="plotly_white")
    return {'data': [trace], 'layout': layout}

@app.callback(
    [Output('train-data-led', 'value'), Output('test-data-led', 'value')],
    [Input('train-test-slider', 'value')]
)
def update_led_display(value):
    # Assuming the 'value' from the slider represents the proportion
    # of the data to be used for training, and that the slider's value
    # ranges from 0 to 1.

    total_data = X_pca.shape[0]
    train_data = int(total_data * value/100)
    test_data = total_data - train_data

    return train_data, test_data


if __name__ == "__main__":
    app.run_server(debug=True)
