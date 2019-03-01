from flask import Flask, abort, request
from flask_restful import Resource, Api
from flask import make_response
import os, sys, json, time, uuid,  re, io, numpy as np
from os import path
from PIL import Image
from flask_cors import CORS, cross_origin
from object_detector_pytorch import ObjDetectorImp,ClassNameImp
import cv2

JSON_MIME_TYPE = 'application/json'
HTML_MIME_TYPE = 'text/html'
PAD = 114

app = Flask(__name__)
api = Api(app)
cors = CORS(app, resources={r"/foo": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'



class CNTK(Resource):
    @cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
    def get(self):
        resp = "<h1> Welcome To VOTT Reviewer Service: <br/> CNTK Endpint</h1>"
        return make_response(resp, 200, {'Content-Type': HTML_MIME_TYPE})

    @app.route('/cntk', methods=['POST'])
    @cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
    def post():
        id = str(uuid.uuid4())
        fname = request.values.get("filename") or (id+".jpg")
        image = Image.open(request.files['image'].stream)

        image_base_name=fname

        image=cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        class_IDs, scores, bounding_boxs,elapsed_obj=ObjDetectorImp.Detect(image)

        vott_classes = {
            ClassNameImp[i]: i
            for i in range(len(ClassNameImp))
        }
       
        json_output_obj = {"classes": vott_classes, "frames": {}}
        regions_list=[]

        for index in range(0,len(bounding_boxs)):
            if ClassNameImp[class_IDs[index]]=='person':
               if (scores[index]>0.4):     
                                        
                     print('Person {}%'.format(scores[index]))
                     regions_list.append({
                          "x1": int(bounding_boxs[index][0]),
                          "y1": int(bounding_boxs[index][1]),
                          "x2": int(bounding_boxs[index][2]),
                          "y2": int(bounding_boxs[index][3]),
                          "class": class_IDs[index]# ClassNameImp[class_IDs[index]]
                          })
            json_output_obj["frames"][image_base_name] = {
                "regions": regions_list
            }

        json_dump = json.dumps(json_output_obj, indent=2)

         
        # print(model_path, file=sys.stdout)
        data = json_dump
        # print(data, file=sys.stdout)
        return make_response(data, 200, {'Content-Type': JSON_MIME_TYPE})


class Home(Resource):
    @cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
    def get(self):
        resp = '<html><head></head><body><h1>Welcome To VOTT Reviewer Service</h1> <br/> <p>Apis available:</p><ul><li>get: /</li><li>post: /cntk</li></ul></body></html>'
        return make_response(resp, 200, {'Content-Type': HTML_MIME_TYPE})


api.add_resource(CNTK, '/cntk')
api.add_resource(Home, '/')


@app.errorhandler(404)
def not_found(e):
    return '', 404
