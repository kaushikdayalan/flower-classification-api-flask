from flask import Flask, render_template, request
from predictor import get_prediction

app = Flask(__name__)

@app.route("/",methods = ["GET","POST"])
def hello_world():
    if request.method == "GET":
        return render_template("index.html") 

    if  request.method == "POST":
        if "file" not in request.files:
            return "file not uploaded !!!"
        
        file = request.files["file"]
        image = file.read()
        flower = get_prediction(image_bytes=image)
        return render_template("result.html", value=flower)
if __name__ == "__main__":
    app.run()