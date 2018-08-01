# -*- coding:utf8 -*-
# !/usr/bin/env python
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
from future.standard_library import install_aliases
install_aliases()

from urllib.parse import urlparse, urlencode
from urllib.request import urlopen, Request
from urllib.error import HTTPError

import json
import os

from flask import Flask
from flask import request
from flask import make_response
from flask import jsonify

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

# ---------------- General Functions needed for parts of the Webhook -----------------------

# ------------------------------ Actual Webhook work starts from here ------------------------------------------

# Flask app should start in global layout
app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(silent=True, force=True)

    print("Request:")
    print(json.dumps(req, indent=4))

    res = processRequest(req)

    res = json.dumps(res, indent=4)
    print("Response:")
    print(res)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    
    return r

#Now the processRequest method is where you'll get most of your work done ;-)
def processRequest(req):
    ClumpThickness = req['ClumpThickness']
    UniformityOfCellSize = req['UniformityOfCellSize']
    UniformityOfCellShape = req['UniformityOfCellShape']
    MarginalAdhesion = req['MarginalAdhesion']
    SingleEpithelialCellSize = req['SingleEpithelialCellSize']
    BareNuclei = req['BareNuclei']
    BlandChromatin = req['BlandChromatin']
    NormalNucleoli = req['NormalNucleoli']
    Mitoses = req['Mitoses']
    values_to_predict = [[ClumpThickness, UniformityOfCellSize, UniformityOfCellShape, MarginalAdhesion, SingleEpithelialCellSize, BareNuclei, BlandChromatin, NormalNucleoli, Mitoses]]
    print("Values to predict:")
    print(values_to_predict)

    tpot_data = pd.read_csv('breastcancer.csv', sep=',', dtype=np.int_)
    features = tpot_data.drop('Class', axis=1).values
    target = tpot_data.drop(['ClumpThickness', 'UniformityOfCellSize', 'UniformityOfCellShape', 'MarginalAdhesion', 'SingleEpithelialCellSize', 'BareNuclei', 'BlandChromatin', 'NormalNucleoli', 'Mitoses'], axis=1).values
    print("Features")
    print(features)
    print("Target")
    print(target)
    exported_pipeline = make_pipeline(
    FastICA(tol=0.7000000000000001),
    KNeighborsClassifier(n_neighbors=56, p=1, weights="distance")
    )

    exported_pipeline.fit(features, target)
    result = exported_pipeline.predict(values_to_predict)

    prediction = str(result[0])
    
    return {"results": prediction}

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))

    print("Starting app on port %d" % port)

    app.run(debug=False, port=port, host='0.0.0.0')