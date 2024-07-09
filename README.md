# Post Market Evaluation Pilot 4

## Description

This repository will contain the files necassary for the post-market evaluation within Pilot 4 using the COPD dataset. Currently, an initial version of the API endopoints has been created to demonstrate the functionality of the module.

## Usage

### Installation

Create a virtual environment using `venv` and install all necessary dependencies using the provided `requirements.txt` file. This can be done using the following commands:

```
python3 -m venv .venv
source .venv/bin/activate`
python3 -m pip install -r requirements.txt
```

### Deploy using Fastapi / Uvicorn

To deploy the application:

```
python3 ./main.py
```

or:

```
uvicorn main:app --reload
```

The API documentation (Swagger UI) can be found at:

```
http://127.0.0.1:8000/docs
```

### (Opt.) Train the adversarial evaluation models

Even though the models used for adversarial evaluation have already been trained and are included in the `models` folder, they can be trained from scratch by running:

```
cd source
python3 ./train_adversarial_models.py --data_path "path/to/data.csv" --test_size 0.3
```
