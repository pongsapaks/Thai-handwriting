import datetime
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from dash_canvas import DashCanvas
from PIL import Image
import numpy as np
import tensorflow as tf
import base64

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True

app.layout = html.Div(
    className="container",
    children=[
        html.H1("Thai Number Handwriting Recognition", className="header"),
        dcc.Upload(
            id='upload-image',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            className="upload",
            multiple=True
        ),
        html.Div(id='output-image-upload'),
        html.H5('Draw inside the canvas to input Thai numbers', className="instruction"),
        html.Div(
            DashCanvas(
                id='canvas_101',
                width=280,
                height=280,
                hide_buttons=['line', 'zoom', 'pan', 'select'],
                goButtonTitle='Recognize',
            ),
            className="canvas"
        ),
        html.Div(id='canvas-output', className="recognized-number"),
        dcc.Slider(0, 100, 1, value=100, marks=None,
               tooltip={"placement": "bottom", "always_visible": True}),
    html.Div(id='recognized-output')  # New Div element for displaying recognized output
    ]
)


def parse_contents(contents, filename, date):
    # Read the image from contents using PIL
    img = Image.open(io.BytesIO(contents))
    
    # Preprocess the image (resize, convert to grayscale, normalize, etc.)
    img = img.resize((28, 28))
    img_gray = img.convert('L')
    img_gray = np.array(img_gray)
    
    # Perform handwriting recognition here with the img_gray data
    recognized_number = "Recognized Number: {}".format(json_data)  # Replace with your recognition logic
    
    # Return the HTML representation of the result
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),
        html.Img(src=contents),
        html.Hr(),
        html.Div([
            html.H4(recognized_number, className="recognized-text"),
            html.Img(src="data:image/png;base64,{}".format(image_to_b64(img_gray))),
        ])
    ])


# def parse_contents(contents, filename, date):
#     # Read the image from contents using PIL
#     img = Image.open(io.BytesIO(contents))
    
#     # Preprocess the image (resize, convert to grayscale, normalize, etc.)
#     img = img.resize((28, 28))
#     img = img.convert('L')
#     img = np.array(img) / 255.0
#     img = img.reshape((1, 28, 28, 1))  # Reshape to match model input shape
    
#     # Make predictions using your trained model
#     predictions = model.predict(img)  # Assuming you have a trained model named 'model'
#     predicted_label = np.argmax(predictions)
    
#     # Return the HTML representation of the result
#     return html.Div([
#         html.H5(filename),
#         html.H6(datetime.datetime.fromtimestamp(date)),
#         html.Img(src=contents),
#         html.Hr(),
#         html.Div(f'Predicted Label: {predicted_label}')
#     ])

@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))

def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)
        ]
        return children


@app.callback(
    Output('canvas-output', 'children'),
    Input('canvas_101', 'json_data')
)
# def recognize_handwriting(json_data):
#     if json_data:
#         # Perform handwriting recognition here with the json_data
#         recognized_number = "Recognized Number: {json_data}"  # Replace with your recognition logic
#         return html.H4(recognized_number, className="recognized-text")
def recognize_handwriting(json_data):
    if json_data:
        # Perform handwriting recognition here with the json_data
        recognized_number = "Recognized Number: {}".format(json_data)  # Replace with your recognition logic
        return html.H4(recognized_number, className="recognized-text")
    else:
        return html.H4("No data provided", className="recognized-text")

def image_to_b64(img):
    _, img_encoded = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    return img_base64


if __name__ == '__main__':
    app.run_server(debug=True)



# import datetime

# from dash import Dash, dcc, html
# from dash.dependencies import Input, Output, State
# from dash_canvas import DashCanvas
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# app = Dash(__name__, external_stylesheets=external_stylesheets)
# app.config.suppress_callback_exceptions = True
# app.layout = html.Div([
#     dcc.Upload(
#         id='upload-image',
#         children=html.Div([
#             'Drag and Drop or ',
#             html.A('Select Files')
#         ]),
#         style={
#             'width': '100%',
#             'height': '28px',
#             'lineHeight': '28px',
#             'borderWidth': '1px',
#             'borderStyle': 'dashed',
#             'borderRadius': '5px',
#             'textAlign': 'center',
#             'margin': '10px'
#         },
#         # Allow multiple files to be uploaded
#         multiple=True
#     ),
#     html.Div(id='output-image-upload'),
#     html.H5('Press down the left mouse button and draw inside the canvas'),
#     DashCanvas(id='canvas_101'),
#     dcc.Slider(0, 100, 1, value=100, marks=None,
#     tooltip={"placement": "bottom", "always_visible": True})
# ])

# def parse_contents(contents, filename, date):
#     return html.Div([
#         html.H5(filename),
#         html.H6(datetime.datetime.fromtimestamp(date)),

#         # HTML images accept base64 encoded strings in the same format
#         # that is supplied by the upload
#         html.Img(src=contents),
#         html.Hr(),
#         html.Div('Raw Content'),
#         html.Pre(contents[0:200] + '...', style={
#             'whiteSpace': 'pre-wrap',
#             'wordBreak': 'break-all'
#         })
#     ])

# @app.callback(Output('output-image-upload', 'children'),
#               Input('upload-image', 'contents'),
#               State('upload-image', 'filename'),
#               State('upload-image', 'last_modified'))
# def update_output(list_of_contents, list_of_names, list_of_dates):
#     if list_of_contents is not None:
#         children = [
#             parse_contents(c, n, d) for c, n, d in
#             zip(list_of_contents, list_of_names, list_of_dates)]
#         return children

# if __name__ == '__main__':
#     app.run_server(debug=True)
# from dash import Dash, html
# from dash_canvas import DashCanvas

# app = Dash(__name__)
# app.config.suppress_callback_exceptions = True

# app.layout = html.Div([
#     html.H5('Press down the left mouse button and draw inside the canvas'),
#     DashCanvas(id='canvas_101')
#     ])


# if __name__ == '__main__':
#     app.run_server(debug=True)

