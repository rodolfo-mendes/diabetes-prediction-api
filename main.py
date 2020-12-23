'''
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org>
'''
import logging
import numpy as np

from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel, Field
from typing import List

log = logging.getLogger(__name__)

class Patient(BaseModel):
    pregnancies: int = Field(..., title='Pregnancies', description='Number of times pregnant', ge=0)
    glucose: int = Field(..., title='Glucose', \
        description='Plasma glucose concentration a 2 hours in a oral glucose tolerance test', ge=0)
    blood_pressure: int = Field(..., title='Blood Pressure', description='Diastolic blood pressure (mm Hg)', ge=0)
    skin_thickness: int = Field(..., title='Skin Thickness', description='Triceps skin fold thickness (mm)', ge=0)
    insulin: int = Field(..., title='Insulin', description='2-Hour serum insulin (muU/ml)', ge=0)
    bmi: float = Field(..., title='BMI', description='Body mass index (kg/m²)', ge=0)
    diabetes_pedigree_function: float = Field(..., title='Diabetes pedigree function',\
        description='Diabetes pedigree function', ge=0)
    age: int = Field(..., title='Age', description='Age (years)', ge=0)

    def as_array(self):
        return [self.pregnancies, self.glucose, self.blood_pressure, self.skin_thickness, self.insulin, self.bmi,\
        self.diabetes_pedigree_function, self.age]

class Prediction(BaseModel):
    diabetes: bool
    diabetes_probability: float

def __predict_batch(p_list: List[Patient]):
    X = np.array([p.as_array() for p in p_list])

    y = diabetes_model.predict(X).astype('bool')
    y_probabilities = diabetes_model.predict_proba(X)[:, 1]

    return [Prediction(diabetes=z, diabetes_probability=z_prob) for z, z_prob in zip(y, y_probabilities)]

app = FastAPI()

diabetes_model = None

@app.on_event('startup')
def init():
    global diabetes_model
    diabetes_model = load('model/diabetes_predictor.joblib')

@app.get('/')
async def root():
    return {"message": "Hello World"}

@app.post("/diabetes/prediction", response_model=Prediction)
async def predict(p: Patient):
    return __predict_batch([p])[0]

@app.post("/diabetes/prediction/batch", response_model=List[Prediction])
async def predict_batch(p_list: List[Patient]):
    return __predict_batch(p_list)

