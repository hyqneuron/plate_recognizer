
def conv_bn_relu(num_in, num_out, kernel_size=3, use_relu=True):
    layers = [('conv', nn.Conv2d(num_in, num_out, kernel_size, padding=(kernel_size-1)/2))]
    if use_relu:
        layers += [ ('bn',   nn.BatchNorm2d(num_out)),
                    ('relu', nn.ReLU(True)) ]
    return nn.Sequential( OrderedDict(layers) )

def deconv_bn_relu(num_in, num_out, kernel_size=3, stride=1, use_relu=True):
    layers = [('deconv', nn.ConvTranspose2d(num_in, num_out, kernel_size, stride=stride, padding=(kernel_size-1)/2))]
    if use_relu:
        layers += [ ('bn',   nn.BatchNorm2d(num_out)),
                    ('relu', nn.ReLU(True)) ]
    return nn.Sequential( OrderedDict(layers) )

def build_conv_layers(bnf):
    global black_white
    stages = OrderedDict([
            ('conv1_1', conv_bn_relu(num_input_cn,   bnf, 3)), #3, 1
            ('conv1_2', conv_bn_relu(bnf, bnf, 3)),     # 5, 1
            ('pool1',   nn.MaxPool2d(2,2)),             # 6, 2

            ('conv2_1', conv_bn_relu(bnf,  bnf*2, 3)),  # 10, 2
            ('conv2_2', conv_bn_relu(bnf*2,bnf*2, 3)),  # 14, 2
            ('pool2',   nn.MaxPool2d(2,2)),             # 16, 4

            ('conv3_1', conv_bn_relu(bnf*2,bnf*4, 3)),  # 24, 4
            ('conv3_2', conv_bn_relu(bnf*4,bnf*4, 3)),  # 32, 4
            ('conv3_3', conv_bn_relu(bnf*4,bnf*4, 3)),  # 40, 4
            ('pool3',   nn.MaxPool2d(2,2)),             # 44, 8

            ('conv4_1', conv_bn_relu(bnf*4,bnf*8, 3)),  # 60, 8
            ('conv4_2', conv_bn_relu(bnf*8,bnf*8, 3)),  # 76, 8
            ('conv4_3', conv_bn_relu(bnf*8,bnf*8, 3)),  # 92, 8
        ])
    return CachedSequential(stages)

class CachedSequential(nn.Sequential):
    def forward(self, input):
        cache_targets = ['pool1', 'pool2', 'conv4_3']
        self.cached_outs = {}
        for name, module in self._modules.iteritems():
            input = module(input)
            if name in cache_targets:
                self.cached_outs[name] = input
        return input

class VGGLikeHeatRegressor(nn.Module):
    def __init__(self, bnf):
        # bnf is base number feature
        super(VGGLikeHeatRegressor, self).__init__()
        self.cnn = build_conv_layers(bnf)
        self.push_2 = conv_bn_relu(bnf*2, bnf*2)
        self.push_4 = conv_bn_relu(bnf*8, bnf*2)
        self.ups_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.ups_2 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.ups_4 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.conv_concat = conv_bn_relu(bnf*5, num_classes, 3, use_relu=False)
        self.softmax = nn.Softmax2d()
    def forward(self, input):
        self.cnn(input)
        out_ups1 = self.ups_1(self.cnn.cached_outs['pool1'])
        out_ups2 = self.ups_2(self.push_2(self.cnn.cached_outs['pool2']))
        out_ups4 = self.ups_4(self.push_4(self.cnn.cached_outs['conv4_3']))
        concat = torch.cat([out_ups1, out_ups2, out_ups4], 1)
        return self.softmax(self.conv_concat(concat))
