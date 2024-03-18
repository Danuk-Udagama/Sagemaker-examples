# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import io
import json
import os
import pickle
import joblib
import signal
import sys
import traceback
import numpy as np

import flask
import pandas as pd

prefix = "/opt/program/"
model_path = os.path.join(prefix, "model")

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.


class ScoringService(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            with open(os.path.join(model_path, "model.joblib"), "rb") as inp:
                cls.model = joblib.load(inp)
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        clf = cls.get_model()
        return clf.predict(input)


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single data item
    """

    data = None

    if flask.request.content_type == "application/json":
        data = flask.request.data.decode("utf-8")
        data = json.loads(data)
        inputVal = data['Inputs']

        # data = io.StringIO(data)
    else:
        return flask.Response(
            response="This predictor only supports JSON data, the provided format {}".format(flask.request.content_type), status=415, mimetype="text/plain"
        )

    # Do the prediction
    predictions = ScoringService.predict(inputVal)


    predictions = predictions.tolist()
    result = {"Outputs":predictions}
    response =json.dumps(result)

    return flask.Response(response=response, status=200, mimetype="application/json")
