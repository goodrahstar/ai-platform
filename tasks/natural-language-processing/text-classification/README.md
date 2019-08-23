# Text Classification

This example demonstrate how to train a deep learning model to perform text classification using MLFlow. All the logs are captures and stores in the `mlruns` folder.


## Folder structure


```.
├── conda.yaml
├── config.json                  # Contains all the config to train the model
├── data
│   └── textdata.csv
├── __init__.py
├── MLproject
├── mlruns
│   └── 0
├── model                        # Definition of the keras model
│   ├── __init__.py
│   └── textcnn_model.py
├── README.md
├── trainer.py                   # Main entry point
└── utils                        # Preprocessing modules 
    ├── __init__.py
    └── textcnn_preprocess.py
```
# Running this Example

To train the model, run the example as a standard MLflow project:

1. Git clone :

    `git clone https://github.com/goodrahstar/ai-platform`
    `cd ai-platform` 
    
3. Modify Config:
If you want to make any modifications on the config parameters which stores the information about the dataset and model hyper parameters. It is available inside `./task/natural-language-processing/text-classification/config.json` .  And looks like this:


```
        {
            "experiment_name": "abc2",
            "file_path": "./data/textdata.csv",
            "input_dim": 22,
            "model_path": "./model/sample.h5",
            "model_type": "textcnn",
            "maxlen": 100,
            "train": true,
            "epochs": 2,
            "batch_size": 10
          }
```

2. Train the model:

    `mlflow run tasks/natural-language-processing/text-classification`

3. Deploy the model:

   `mlflow models serve --model-uri runs:///18fa522576934ac08a989b1cba6af124/model --port 12345`


