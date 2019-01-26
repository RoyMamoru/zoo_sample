from django.db import models

class ZooCollection(models.Model):
    # ログインUserの情報を保存するテーブル
    user_id = models.IntegerField(default=0) # 見つけたUserのIDを保存
    animal_id = models.IntegerField(default=0) # 見つけた動物のIDを保存

    def __str__(self):
        return '%d, %d' % (self.user_id, self.animal_id)

class AnimalInfo(models.Model):
    # 動物の情報を保存するテーブル
    id = models.AutoField(primary_key=True)
    animal_id = models.IntegerField(default=0) # 動物のID
    animal_name = models.CharField(max_length=50, default='') # 動物の名前
    animal_title = models.CharField(max_length=300, default='') # 動物解説のタイトル
    animal_disc = models.TextField(default='') # 動物解説の詳細

    def __str__(self):
        return '%d, %s, %s, %s' % (self.animal_id, self.animal_name, self.animal_title, self.animal_disc)
