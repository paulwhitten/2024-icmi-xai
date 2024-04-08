from typing import Union
from fastapi import FastAPI, APIRouter
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from transform_parallel import get_image_from_list, image_data_to_list, get_transforms
from skimage import filters
from skimage.measure import regionprops
from calculate_sample import CalcObject
import numpy as np
#import uvicorn
import logging
import json

def center_of_mass(data):
    image = get_image_from_list(data)
    threshold_value = filters.threshold_otsu(image)
    labeled_foreground = (image > threshold_value).astype(int)
    properties = regionprops(labeled_foreground, image)
    center_of_mass = properties[0].centroid
    weighted_center_of_mass = properties[0].weighted_centroid
    print("COM:", center_of_mass)
    return center_of_mass

def translate_img(img, x_trans, y_trans):
    im_sh = img.shape
    raw = np.zeros((im_sh[0], im_sh[1]), np.uint8)
    for x in range(im_sh[0]):
        for y in range(im_sh[1]):
            x_pos = x + x_trans
            y_pos = y + y_trans
            if x_pos >= 0 and x_pos < im_sh[0] and y_pos >= 0 and y_pos < im_sh[1]:
                raw[x_pos,y_pos] = img[x,y]
    return raw

# setup loggers
#logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)
FORMAT = "%(levelname)s:  %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

logging.debug('This message should appear on the console')

#CalcObject("models_svm_mnist", "kb_svm_mnist")
#CalcObject("models/mnist/mlp", "kb/mnist/mlp")
calc = CalcObject("models/mnist/mlp", "kb/mnist/mlp")

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

            com = center_of_mass(image_data)
            img = get_image_from_list(image_data)
            centered = translate_img(img, int(13.0-com[0]), int(13.0-com[1]))
            centered_data = image_data_to_list(centered)
            
            # transform
            raw, thresh, skel, fill, corner, ellipse, circle, ellipse_circle, skel_fill, crossing, endpoint, line, chull = get_transforms(centered_data)
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