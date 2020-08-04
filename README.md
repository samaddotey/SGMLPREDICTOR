# sg-predict

### Tech

sg-predict is an ML application that suggests SG Market products based on Industry and Key Process input


### Installation

Install the dependencies and devDependencies and start the server.

```sh
$ pip install requests

$ pip install flask

$ pip install flask_cors

$ pip install install numpy

$ pip install sklearn

$ pip install pandas

$ pip install aikit
```


### Development
Pre-processed company dataset from Kaggle
Put the data through transformer (Numerical Encoder)
Split data up into training and test (70:30)
Trained ML model (Logistic Regression)
Saved model (Pickle)

#### Run Application Instructions
To run webserver to launch application:
```sh
$ python server.py
```
Launch Browser and navigate to http://localhost:3001/

NOTE: The train model command does not need to be run at all but added for explantory purposes

#### To Train Model
```sh
$ python ML.py
```


