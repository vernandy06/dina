from flask import Flask, render_template, request, url_for
from model import Model
import os
import json
import pandas as pd
from werkzeug.utils import secure_filename

app = Flask(__name__)
dataforecast = None

# Enable debugging mode
app.config["DEBUG"] = True

# Upload folder
UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello():
    return render_template("index.html")

@app.route("/predicted", methods=['POST'])
def pred():
    global dataforecast
    if request.method == "POST":
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(uploaded_file.filename))
            uploaded_file.save(file_path)

            model = Model(file_path)
            df = model.data()
            model.pairplot()
            model.plotcase()
            model.corr_plot()
            model.getDataResult()
            ev = model.evaluate()
            res = model.getModel()
            model.movingaverage()
            model.forecast()

            # Parsing the DataFrame in JSON format
            json_records = df.to_json(orient='records')
            data = json.loads(json_records)

            dataforecast = json.loads(model.forecastData().to_json(orient='records'))
            datas = [data, ev, res]

            return render_template("hasil.html", data=datas)

@app.route("/forecasting")
def forecasting():
    global dataforecast
    return render_template("forecasting.html", data=dataforecast)

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
