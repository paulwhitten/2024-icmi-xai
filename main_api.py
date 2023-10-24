from typing import Union
from fastapi import FastAPI, APIRouter
from fastapi import Request
from fastapi.staticfiles import StaticFiles
#import uvicorn
import logging

# setup loggers
#logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)
FORMAT = "%(levelname)s:  %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

logging.debug('This message should appear on the console')

app = FastAPI()
router = APIRouter()

app.mount("/static", StaticFiles(directory="static"), name="static")

@router.get("/")
def router_root():
    logger.debug("router root")
    return {"Hello": "Router Root"}

app.include_router(router, prefix="/router")

@app.get("/")
def read_root():
    logger.debug("Root log")
    return {"Hello": "World"}

@app.post("/submit")
async def submit(request: Request):
    logger.debug("Submit log")
    content_type = request.headers.get('Content-Type')
    if content_type is None:
        logger.error("content error")
        return 'No Content-Type provided.'
    elif content_type == 'application/json':
        try:
            logger.debug("submit json processing")
            json = await request.body()
            # TODO: process the data
            logger.debug(json)
            return json
        except JSONDecodeError:
            logger.error("JSON error")
            return 'Invalid JSON data.'
    else:
        logger.error("content type not supported")
        logger.error(content_type)
        return 'Content-Type not supported.'

    #b = await request.body()
    #print(b)
    ## TODO process

    #return {"json": {}}

#if __name__ == '__main__':
#    uvicorn.run("main_api:app", port=8000)