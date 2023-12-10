# 2023-svm-xai

## process

TODO document the process

## Web api

A web api is implemented in `main_api.py` using fast api.

Start by opening the Python virtual environment and then running:

```sh
uvicorn main_api:app --reload
```

The API will serve:

- [An http web interface at http://127.0.0.1:8000/static/web_draw.html](http://127.0.0.1:8000/static/web_draw.html)
- POST to http://127.0.0.1:8000//submit that the page submits JSON data to for recognition from the webpage

