# 2024-icmi-xai

This set of programs is for TODO provide a link to where this will be published

## Process

TODO: refine the documentation of the process

1. Create a folder for transformed data
1. Transform data using the program `transform_parallel.py` into the folder for transforms
1. Create a folder to store the ML models
1. Train models using the program `train_svm_transforms.py` (support vector machines) or `train_nn_transforms.py` (multi-layer perceptrons) storing in the appropriate folder
1. Creating a kb folder.
1. Build the knowledgebase (kb) by running the program `build_kb.py` with output of the kb folder
1. Run copy <model_foler>/*.json <kb_folder>/
1. Run the program `process_kb_conf_matrix.py` on the kb folder to generate confusion matrices and metrics
1. Run the program `node process_kb.js` against the kb folder to get results against test data with various effectiveness metrics and compare mixed versus only explainable via command line options.

## Web api

A web api is implemented in `main_api.py` using fast api.

Start by opening the Python virtual environment and then running:

```sh
uvicorn main_api:app --reload
```

The API will serve:

- [An http web interface at http://127.0.0.1:8000/static/web_draw.html](http://127.0.0.1:8000/static/web_draw.html)
- POST to http://127.0.0.1:8000//submit that the page submits JSON data to for recognition from the webpage
