from django.shortcuts import render, redirect

# DBのインポート
from .models import ZooCollection, AnimalInfo
import csv
# 画像ファイルUploadの際に使用
from django.conf import settings
from django.core.files.storage import FileSystemStorage
# 推論に必要なモジュールの読み込み
import numpy as np
import cv2
import chainer
import chainer.links as L
import chainer.functions as F
from .trained_model.trained_model import GoogleNetModel
from chainer.links.model.vision.googlenet import prepare

# モデルの読み込み
model = L.Classifier(GoogleNetModel())
chainer.serializers.load_npz('zoo_app/trained_model/model_gnet_finetune.npz', model)

# DB内にデータがない場合に動物データを登録したDBの作成
animalsinfo = list(AnimalInfo.objects.all())
if not animalsinfo:
    with open('zoo_app/static/data/animal_data.csv', 'r') as f:
        data = csv.reader(f)
        data = [row for row in data]
    for i, _data in enumerate(data[1:]):
        ani_name, ani_title, ani_disc = _data
        aniinfo = AnimalInfo(animal_id=i, animal_name=ani_name, animal_title=ani_title, animal_disc=ani_disc)
        aniinfo.save()

def classify(request):
        # 画像データを取得＆保存
    if request.method == 'POST' and request.FILES['predict_img']:
        predict_img = request.FILES['predict_img']
        fs = FileSystemStorage()
        filename = fs.save(predict_img.name, predict_img)
        uploaded_file_url = fs.url(filename)

        # 推論処理
        img = cv2.cvtColor(cv2.imread(uploaded_file_url), cv2.COLOR_BGR2RGB)
        x = prepare(img)
        y = model.predictor(np.array([x]))
        y_proba = F.softmax(y).data
        y_pre = np.argmax(y_proba, axis=1)[0]
        proba = round(y_proba[0][y_pre] * 100, 2)

        # AnimalInfoのDBから必要な情報の取得
        animal_info = AnimalInfo.objects.filter(animal_id=y_pre)

        # ZooCollectionに情報を保存
        current_user = request.user
        if not list(ZooCollection.objects.filter(user_id=current_user.id, animal_id=y_pre)):
            user_info = ZooCollection(user_id=current_user.id, animal_id=y_pre)
            user_info.save()

        return render(request, 'zoo_app/classify.html',{'uploaded_file_url':uploaded_file_url, 'animal_info':animal_info, 'proba':proba})

    return render(request, 'zoo_app/classify.html', {})
