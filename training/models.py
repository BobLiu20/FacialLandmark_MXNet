import sys

from mxnet.gluon import nn
from mxnet import gluon

class fanet8ss_inference(gluon.Block):
    def __init__(self, num_label, **kwargs):
        super(fanet8ss_inference, self).__init__()
        self.num_label = num_label
        self.kwargs = kwargs
        with self.name_scope():
            self.models = self.net()

    def forward(self, x):
        if self.kwargs.get("train_angle", False):
            public_block, landmark_block, angle_block = self.models
        else:
            public_block, landmark_block = self.models
        x = public_block(x)
        if self.kwargs.get("train_angle", False):
            return landmark_block(x), angle_block(x)
        else:
            return landmark_block(x)

    def net(self):
        model = nn.Sequential()
        # public block
        public_block = nn.HybridSequential()
        with public_block.name_scope():
            public_block.add(
                nn.Conv2D(16, kernel_size=3, strides=2, activation="relu"),

                nn.Conv2D(32, kernel_size=3, strides=1, activation="relu"),
                nn.Conv2D(32, kernel_size=3, strides=2, activation="relu"),

                nn.Conv2D(32, kernel_size=3, strides=1, activation="relu"),
                nn.Conv2D(32, kernel_size=3, strides=2, activation="relu"),

                nn.Conv2D(64, kernel_size=3, strides=1, activation="relu"),
                nn.Conv2D(64, kernel_size=3, strides=2, activation="relu"),

                nn.Conv2D(128, kernel_size=3, strides=1, activation="relu"),
                nn.MaxPool2D(pool_size=2, strides=2),
            )
        model.add(public_block)
        # landmark_block
        landmark_block = nn.HybridSequential()
        with landmark_block.name_scope():
            landmark_block.add(
                nn.Dense(512, flatten=True),
                nn.Dense(self.num_label, activation="sigmoid")
            )
        model.add(landmark_block)
        # angle_block
        if self.kwargs.get("train_angle", False):
            angle_block = nn.HybridSequential()
            with angle_block.name_scope():
                angle_block.add(
                    nn.Dense(512, flatten=True),
                    nn.Dense(3, activation="sigmoid")
                )
            model.add(angle_block)
        return model

# init
def init(model, num_label, **kwargs):
    func = getattr(sys.modules["models"], model)
    return func(num_label, **kwargs)

if __name__ == "__main__":
    import models
    net = models.init("fanet8ss_inference", num_label=72*2)
    net.initialize()
    print net
    from mxnet import nd
    test_data = nd.random.uniform(shape=(1,3,128,128))
    landmark, angle = net(test_data)
    print landmark.shape, angle.shape

