from flask import Flask, render_template, request
import vision
import criticize
import time

app = Flask(__name__)


@app.route("/")
def index(app_name="TerribleHack"):
    return render_template("index.html", app=app_name)


@app.route("/analyze", methods=["POST"])
def process():
    uri = request.form["url"]
    labels = [l.description for l in vision.label_img(uri)]
    # labels = ["happiness", "friends", "food"]
    # interps = ["wasting time", "wasting money"]
    interps = criticize.interpret_labels(labels)
    time.sleep(5)
    return render_template(
        "analyze.html", url_submission=uri, labels=labels, interps=interps
    )


# dumb page to test out layouts n stuff
@app.route("/dummy")
def dummy(var="Example"):
    return render_template("dummy.html", variable=var)


if __name__ == "__main__":
    app.run(debug=True)
