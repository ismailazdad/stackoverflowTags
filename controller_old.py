import sys
from flask import Flask, jsonify, request, render_template, Config
import torch
from PIL import Image
import datetime
from datetime import datetime

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/hello')
def hello():
    return render_template('index.html', utc_dt=datetime.utcnow())


@app.route('/testModeleImage', methods=['POST'])
def testModeleImage():
    if app.config.get('env') == 'prod':
        CPU = True
        PATH = '/root/flask/best.pt'
        PATH_IMAGE = '/root/flask/static/img/detect/'
        FORCE_RELOAD = False
    else:
        CPU = False
        PATH = '/media/isma/travail/algoscope/train_result/hemato_result/exp_2021100820_number_batch_10_image_size_608_epochs_300_config_file_hemato2.yaml_weight_yolov5x.pt/exp/weights/best.pt'
        PATH_IMAGE = '/media/isma/travail/algoscope/flask/static/img/detect/'
        FORCE_RELOAD = True

    imagefile = request.files['file']
    # selectAllClasses = request.form.get("all")
    # platelets = request.form.get("Platelets")
    # rbc = request.form.get("RBC")
    # wbc = request.form.get("WBC")
    # class_mapping = {'Platelets': 0, 'RBC': 1, 'WBC': 2}
    class_label = {'Platelets': 'Plaquette(s):', 'RBC': 'Globule(s) rouge:', 'WBC': 'globule(s) blanc:'}
    if imagefile.filename == '':
        return "no file detected"
    else:
        model = torch.hub.load('ultralytics/yolov5', 'custom',
                               path=PATH,
                               # force_reload=True)
                               force_reload=FORCE_RELOAD)
        if CPU:
            model = model.cpu()

        imgtodetect = Image.open(imagefile)
        width, height = imgtodetect.size
        imgs = [imgtodetect]  # batch of images

        # classes filter
        result = [1]
        model.classes = result
        # if selectAllClasses == 'false':
        #     if platelets == 'true':
        #         result.append(class_mapping['Platelets'])
        #     if rbc == 'true':
        #         result.append(class_mapping['RBC'])
        #     if wbc == 'true':
        #         result.append(class_mapping['WBC'])
        #     model.classes = result

        # Inference
        results = model(imgs, size=width)  # includes NMS
        # Yolo Panda Results
        # results.print()
        # results.save()  # or .show()
        # get summary results for all classes detected
        # results.pandas().xyxy[0].groupby(["name"], as_index=True)["name"].count().to_string()
        # results.pandas().xyxy[0].groupby(["name"], as_index=True)["name","class"].count().to_html()
        # results.xyxy[0]  # img1 predictions (tensor)
        # results.pandas().xyxy[0]

        # render html
        results.render()  # updates results.imgs with boxes and labels
        for img in results.imgs:
            # save image result copy to server
            img_base64 = Image.fromarray(img)
            filename = str(datetime.today().strftime('%Y%m%d%H%M%S_') + imagefile.filename)
            img_base64.save(PATH_IMAGE + filename)

            captionresult = str(results.pandas().xyxy[0]
                                .groupby(["name"], as_index=True)["name"]
                                .count()
                                .to_string() \
                                .replace("name\n", "") \
                                .replace("\n", "\t") \
                                .replace("Platelets", class_label["Platelets"]) \
                                .replace("RBC", class_label["RBC"]) \
                                .replace("WBC", class_label["WBC"]))

            print(captionresult)
            # return json result
            t = {
                'errorkeys': [],
                'initialPreview': [
                    '<img style="width:auto;height:auto;max-width:100%;max-height:100%;"  src="/static/img/detect/' + filename + '" alt="img_data" id="imgslot">'
                ],
                'initialPreviewConfig': [
                    {'downloadUrl': '/static/img/detect/' + filename, 'caption': captionresult}
                ],
                'initialPreviewThumbTags': [],
                'append': 'true',
                'fileActionSettings': {
                    'showZoom': 'true',
                    'zoomTitle': "plein écran",
                    'showUpload': 'true',
                    'showRemove': 'true',
                    'showDownload': 'true',
                    'downloadTitle': 'telecharger la detection'
                },
                'forceIframeTransport': 'false',
                'initialPreviewAsData': 'false',
                'initialPreviewFileType': 'image',
                'allowedFileTypes': ["image"],
                'browseClass': "btn btn-info",
                'mainClass': "d-grid",
                'showCaption': 'true',
                'showRemove': 'false',
                'showZoom': 'false',
                'showClose': 'false',
                'showBrowse': 'true',
                'showUpload': 'false',
                'showUploadedThumbs': 'false',
                'showPreview': 'true'
            }

            return jsonify(t)


if __name__ == "__main__":
    app.config['env'] = sys.argv[1]
    if app.config.get('env') == 'prod':
        app.run(host='178.170.47.69', port=5000)
    else:
        app.run(debug=True)
    # app.run(host='178.170.47.69', port=5000)
