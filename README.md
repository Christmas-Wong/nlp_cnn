# Chinese Text Classification with CNN

## How Run this Project
- First, run the command below in your conda env:
```text
pip install -r requirements.txt
```
- Second, Edit the [nlp_cnn/run.sh] File
- Third, Edit the [nlp_cnn/data/config.yml] File
- Forth, run command below:

if your system is linux
```text
bash run.sh
```

if your system is windows
```text
python python_runner.py
```

## Introduction
```text
.
├── README.md
├── data
│   ├── config.yml
│   ├── eval.json
│   ├── runtime.log
│   ├── test.json
│   ├── text_cnn.pt
│   └── train.json
├── finish_model.pkl
├── main.py
├── requirements.txt
├── source
│   ├── __init__.py
│   ├── __pycache__
│   │   └── __init__.cpython-38.pyc
│   ├── core
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   └── util.cpython-38.pyc
│   │   ├── args.py
│   │   ├── data.py
│   │   ├── trainer.py
│   │   └── util.py
│   ├── embedding
│   │   ├── __init__.py
│   │   └── word2vec.py
│   ├── io
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   ├── reader.cpython-38.pyc
│   │   │   └── writer.cpython-38.pyc
│   │   ├── model_saver.py
│   │   ├── reader.py
│   │   └── writer.py
│   ├── model
│   │   ├── __init__.py
│   │   └── text_cnn.py
│   └── pipeline
│       ├── __init__.py
│       └── classification.py
└── test.py

```

## Reference