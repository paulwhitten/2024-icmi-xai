from typing import Union
from fastapi import FastAPI
from fastapi import Request

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/submit")
async def submit(request: Request):
    content_type = request.headers.get('Content-Type')
    if content_type is None:
        return 'No Content-Type provided.'
    elif content_type == 'application/json':
        try:
            json = await request.json()
            return json
        except JSONDecodeError: 
            return 'Invalid JSON data.'
    else:
        return 'Content-Type not supported.'

    #b = await request.body()
    #print(b)
    ## TODO process

    #return {"json": {}}