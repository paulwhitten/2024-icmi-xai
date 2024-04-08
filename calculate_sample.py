import json
import pickle
import argparse
import numpy as np
from transform_parallel import TransformNames
from io import BytesIO
import imageio
import base64

# TODO remove skel-fill
# TODO add ability for mixed explainability 

#raw, thresh, skel, fill, corners, ellipse, circle, ellipse_circle, skel_fill, crossings, endpoints, lines, chull

class CalcObject:
    
    def __init__(self, model_folder, kb_folder):
        self.models = {}
        for n in TransformNames:
            model = pickle.load(open(model_folder + "/" + n + ".model", 'rb'))
            self.models[n]= model
        f = open(kb_folder + "/" + "kb.json")
        self.kb = json.load(f)
        f.close()
        self.max_label = max(self.kb["labels"])
    
    def get_result(self, raw, thresh, skel, fill, corner, ellipse, circle, ellipse_circle, skel_fill, crossing, endpoint, line, chull):
        count = 0

        vote_tally = []
        for i in range(self.max_label + 1):
            vote_tally.append({"class": i, "value": 0.0, "explainability": 0.0, "attributions": []})

        for n in TransformNames:
            match n:
                case "raw":
                    img = raw
                case "thresh":
                    img = thresh
                case "skel":
                    img = skel
                case "fill":
                    img = fill
                case "corner":
                    img = corner
                case "ellipse":
                    img = ellipse
                case "circle":
                    img = circle
                case "ellipse-circle":
                    img = ellipse_circle
                case "skel-fill":
                    img = skel_fill
                case "crossing":
                    img = crossing
                case "endpoint":
                    img = endpoint
                case "line":
                    img = line
                case "chull":
                    img = chull
                case _:
                    print("No match for:", n)

            if n != "skel-fill" and n != "thresh": # raw
                print("property:", n)
                # convert img to normalized
                i = np.array(img).flatten().astype('float32')/255

                # was predict([i])
                pred = self.models[n].predict_proba([i])
                class_pred = np.argmax(pred[0])
                print("prediction probability:", pred[0], ", prediction:", class_pred) #pred[0]
 
                #eff = self.kb["stats"][n]["stats"][pred[0]]["accuracy"] * self.kb["stats"][n]["stats"][pred[0]]["sensitivity"] * self.kb["stats"][n]["stats"][pred[0]]["specificity"] * self.kb["stats"][n]["stats"][pred[0]]["precision"]
                eff = self.kb["stats"][n]["stats"][class_pred]["t_product"]

                print("transform:", n, ", prediction:", class_pred, ", effectiveness:", eff, ", probability:", pred[0][class_pred])

                buf = BytesIO()
                imageio.imwrite(buf, img, format='png')
                b64_img = base64.b64encode(buf.getbuffer()).decode('utf-8')
                print("base 64 image:", b64_img)

                # TODO also use probability
                vote_tally[class_pred]["value"] += eff * pred[0][class_pred]
                if n != "raw":
                    vote_tally[class_pred]["explainability"] += eff * pred[0][class_pred]
                vote_tally[class_pred]["attributions"].append({"name": n, "effectiveness": eff, "image": b64_img}) #img.flatten().tolist()

            #self.kb.stats[n].stats[pred[0]].accuracy, self.kb.stats[n].stats[pred[0]].sensitivity,
            #self.kb.stats[n].stats[pred[0]].specificity, self.kb.stats[n].stats[pred[0]].precision
        print("tally:", vote_tally)
        return vote_tally


# TransformNames = [
#     "crossing",
#     "endpoint",
#     "fill",
#     "skel-fill",
#     "skel",
#     "thresh",
#     "line",
#     "ellipse",
#     "circle",
#     "ellipse-circle",
#     "chull",
#     "raw",
#     "corner"
# ]