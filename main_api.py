from typing import Union
from fastapi import FastAPI, APIRouter
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from transform_parallel import get_transforms
from calculate_sample import CalcObject
#import uvicorn
import logging
import json

# setup loggers
#logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)
FORMAT = "%(levelname)s:  %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

logging.debug('This message should appear on the console')

calc = CalcObject("models_svm_mnist", "kb_svm_mnist")

app = FastAPI()
router = APIRouter()

# static serving of files such as web page
app.mount("/static", StaticFiles(directory="static"), name="static")

# hello world example for routing root api path
@router.get("/")
def router_root():
    logger.debug("router root")
    return {"Hello": "Router Root"}

app.include_router(router, prefix="/router")

@app.get("/")
def read_root():
    logger.debug("Root log")
    return {"Hello": "World"}

# submit post for the rendered image
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
            body_data = await request.body()
            #logger.debug(body_data)
            image_data = json.loads(body_data) # an array representing the image
            #logger.debug(image_data)
            
            # transform
            raw, thresh, skel, fill, corner, ellipse, circle, ellipse_circle, skel_fill, crossing, endpoint, line, chull = get_transforms(image_data)
            res = calc.get_result(raw, thresh, skel, fill, corner, ellipse, circle, ellipse_circle, skel_fill, crossing, endpoint, line, chull)
            # feed the data into the various models to get voted
            # calculate the results based on kb effectiveness
            # assemble a response with explainability

            return json.dumps(res)
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

#uvicorn main_api:app --reload