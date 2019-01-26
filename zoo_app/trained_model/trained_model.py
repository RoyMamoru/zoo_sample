import chainer
import chainer.links as L

class GoogleNetModel(chainer.Chain):

    def __init__(self, n_out=20):
        super().__init__( )
        with self.init_scope():
            self.base = L.GoogLeNet()
            self.fc = L.Linear(None, n_out)

    def __call__(self, x):
        h = self.base(x, layers=['pool5'])
        h = self.fc(h['pool5'])
        return h
