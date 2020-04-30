from paddle.fluid.dygraph import Conv2D, Layer, Pool2D, Linear, Sequential, to_variable
from paddle.fluid.layers import flatten

import paddle.fluid as fluid
import numpy as np
import copy

class LinConPoo(Layer):

    def __init__(self, sequence_list):
        '''
        @Brief
            可用于自定义常用网络结构，将自定义的网络结构列表传入，返回Layer模型
            实际上该类是用于`Conv2D`, `Pool2D`, `Linear`的排列组合

        @Parameters
            sequence_list : 自定义网络结构列表, 列表每一元素为字典或列表, 指定每一层的参数

        @Return
            返回自定义的网络模型

        @Examples
        ------------
        >>> # 可以直接用来搭建VGGNet:

        >>> VGG_list_part1 = [

            {'type':Conv2D, 'num_channels': 3, 'num_filters':64, 'filter_size':3, 'stride':1, 'padding':1, 'act':'relu', 'bias_attr':True},
            {'type':Conv2D, 'num_channels':64, 'num_filters':64, 'filter_size':3, 'stride':1, 'padding':1, 'act':'relu', 'bias_attr':True},

            {'type':Pool2D, 'pool_size':2,     'pool_type':'max',    'pool_stride':2,         'global_pooling':False},

            {'type':Conv2D, 'num_channels':64,  'num_filters':128, 'filter_size':3, 'stride':1, 'padding':1, 'act':'relu', 'bias_attr':True},
            {'type':Conv2D, 'num_channels':128, 'num_filters':128, 'filter_size':3, 'stride':1, 'padding':1, 'act':'relu', 'bias_attr':True},

            {'type':Pool2D, 'pool_size':2,     'pool_type':'max',    'pool_stride':2,         'global_pooling':False},

            {'type':Conv2D, 'num_channels':128, 'num_filters':256, 'filter_size':3, 'stride':1, 'padding':1, 'act':'relu', 'bias_attr':True},
            {'type':Conv2D, 'num_channels':256, 'num_filters':256, 'filter_size':3, 'stride':1, 'padding':1, 'act':'relu', 'bias_attr':True},
            {'type':Conv2D, 'num_channels':256, 'num_filters':256, 'filter_size':3, 'stride':1, 'padding':1, 'act':'relu', 'bias_attr':True},

            {'type':Pool2D, 'pool_size':2,     'pool_type':'max',    'pool_stride':2,         'global_pooling':False},

            {'type':Conv2D, 'num_channels':256, 'num_filters':512, 'filter_size':3, 'stride':1, 'padding':1, 'act':'relu', 'bias_attr':True},
            {'type':Conv2D, 'num_channels':512, 'num_filters':512, 'filter_size':3, 'stride':1, 'padding':1, 'act':'relu', 'bias_attr':True},
            {'type':Conv2D, 'num_channels':512, 'num_filters':512, 'filter_size':3, 'stride':1, 'padding':1, 'act':'relu', 'bias_attr':True},

            {'type':Pool2D, 'pool_size':2,     'pool_type':'max',    'pool_stride':2,         'global_pooling':False},

            {'type':Conv2D, 'num_channels':512, 'num_filters':512, 'filter_size':3, 'stride':1, 'padding':1, 'act':'relu', 'bias_attr':True},
            {'type':Conv2D, 'num_channels':512, 'num_filters':512, 'filter_size':3, 'stride':1, 'padding':1, 'act':'relu', 'bias_attr':True},
            {'type':Conv2D, 'num_channels':512, 'num_filters':512, 'filter_size':3, 'stride':1, 'padding':1, 'act':'relu', 'bias_attr':True},

            {'type':Pool2D, 'pool_size':2,     'pool_type':'max',    'pool_stride':2,         'global_pooling':False},

        ]

        >>> VGG_list_part2 = [

            {'type':Linear, 'input_dim': 512*7*7, 'output_dim':4096, 'act':'relu', 'bias_attr':True},
            {'type':Linear, 'input_dim':4096,     'output_dim':4096, 'act':'relu', 'bias_attr':True},
            {'type':Linear, 'input_dim':4096,     'output_dim':64,   'act':'relu', 'bias_attr':True},
        ]

        在当前paddle版本1.7中, `paddle.fluid.layers.flatten`无法加入至`paddle.fluid.dygraph.Sequential`中, 所以我们将VGGNet以flatten为界拆成两部分

        >>> VGG_part1 = LinConPoo(VGG_list_part1)
        >>> VGG_part2 = LinConPoo(VGG_list_part2)

        >>> import numpy as np

        >>> data = np.ones(shape=(8, 3, 224, 224), dtype=np.float32) # 将该data视作图像数据

        >>> with fluid.dygraph.guard():

                data = to_variable(data)
                x = VGG_part1(data, True)
                x = fluid.layers.flatten(x)
                x = VGG_part2(x, True)
                print(x.numpy().shape)

        >>> # 以上是手动搭建VGG16, 也可以直接调用VGG16类, 该类在`LinConPoo`类上进行封装
        '''

        super(LinConPoo, self).__init__()
        self.__sequence_list = copy.deepcopy(sequence_list)

        # 参数有效检验
        if not isinstance(self.__sequence_list, list): raise ValueError('参数`sequence_list`必须为列表')

        # 每一层模型序列
        self._layers_squence = Sequential()
        self._layers_list = []

        LAYLIST = [Conv2D, Linear, Pool2D]
        for i, layer_arg in enumerate(self.__sequence_list):

            # 不改变原来字典或者列表的值
            # layer_arg = layer_arg.copy()

            # 每一层传入的有可能是列表，也有可能是字典
            if isinstance(layer_arg, dict):

                layer_class = layer_arg.pop('type')

                if not layer_class in LAYLIST:
                    # 进行有效性检验
                    raise KeyError("sequence_list中, 每一层的类型必须在`[Conv2D, Linear, Pool2D]`中")

                # 实例化该层对象
                layer_obj = layer_class(**layer_arg)


            elif isinstance(layer_arg, list):

                layer_class = layer_arg.pop(0)

                if not layer_class in LAYLIST:
                    # 进行有效性检验
                    raise KeyError("sequence_list中, 每一层的类型必须在`[Conv2D, Linear, Pool2D]`中")

                # 实例化该层对象
                layer_obj = layer_class(*layer_arg)


            else:
                raise ValueError("sequence_list中, 每一个元素必须是列表或字典")


            # 指定该层的名字
            layer_name = layer_class.__name__ + str(i)


            # 将每一层添加到 `self._layers_list` 中
            self._layers_list.append((layer_name, layer_obj))

            self._layers_squence.add_sublayer(*(layer_name, layer_obj))

        self._layers_squence = Sequential(*self._layers_list)


    def forward(self, inputs, show_shape=False):
        '''
        @Parameters :
            inputs     :   原始数据
            show_shape :   是否显示每一步的shape, 调试时使用
        '''

        if show_shape:

            x = inputs
            for op in self._layers_list:
                x = op[1](x)
                print(op[0], '\t', x.shape)
            return x

        return self._layers_squence(inputs)


class VGG(fluid.dygraph.Layer):


    def __init__(self, input_channel_num=3, out_dim=2, VGG_part_list1=None, VGG_part_list2=None):
        '''
        @Brief
            用于创建VGG模型网络

        @Parameters
            input_channel_num : VGG网络的输入通道数, 默认为3
            out_dim           : VGG网络的输出维数, 默认为2, 即默认做二分类问题
            VGG_part1         : 自定义网络结构列表, 列表每一元素为字典或列表, 指定每一层的参数, 该部分为`flatten`之前的部分
            VGG_part2         : 与 `VGG_part1` 相同, 该部分为`flatten`之后的部分

        @Return
            默认情况下，返回标准VGG16网络, 最后的全连接层输出维度为2, 即默认做二分类问题

        @Examples
        ------------
        >>> # 创建VGG16模型对象, 做二分类问题
        >>> vgg = VGG()

        >>> import numpy as np
        >>> data = np.ones(shape=(8, 3, 224, 224), dtype=np.float32) # 喂入VGG16的数据要求被resize为(224, 244)
        >>> with fluid.dygraph.guard():

                data = paddle.fluid.dygraph.to_variable(data) # 转化为paddle的数据
                y = vgg(data)
                print(y.numpy().shape)

        ----------------------------------------

        >>> # 如果我们觉得 `VGG_list_part2`太大了, 同时我们要做5分类, 我们可以更改网络结构, 同时将最后一层的`output_dim`设置为5:
        >>> VGG_list_part2 = [

            {'type':Linear, 'input_dim': 512*7*7, 'output_dim':4096, 'act':'relu', 'bias_attr':True},
            {'type':Linear, 'input_dim':4096,     'output_dim':5,    'act':'relu', 'bias_attr':True},
        ]
        >>> vgg = VGG(VGG_list_part2=VGG_list_part2) # 直接将 `VGG_list_part2` 传入即可
        '''


        super(VGG, self).__init__()

        # 以下 `VGG_list_part1`和`VGG_list_part2`是VGG16二分类的默认结构
        if VGG_part_list1 is None:


            self.VGG_part_list1 = [

                {'type':Conv2D, 'num_channels': input_channel_num, 'num_filters':64, 'filter_size':3, 'stride':1, 'padding':1, 'act':'relu', 'bias_attr':True},
                {'type':Conv2D, 'num_channels':64, 'num_filters':64, 'filter_size':3, 'stride':1, 'padding':1, 'act':'relu', 'bias_attr':True},

                {'type':Pool2D, 'pool_size':2,     'pool_type':'max',    'pool_stride':2,         'global_pooling':False},

                {'type':Conv2D, 'num_channels':64,  'num_filters':128, 'filter_size':3, 'stride':1, 'padding':1, 'act':'relu', 'bias_attr':True},
                {'type':Conv2D, 'num_channels':128, 'num_filters':128, 'filter_size':3, 'stride':1, 'padding':1, 'act':'relu', 'bias_attr':True},

                {'type':Pool2D, 'pool_size':2,     'pool_type':'max',    'pool_stride':2,         'global_pooling':False},

                {'type':Conv2D, 'num_channels':128, 'num_filters':256, 'filter_size':3, 'stride':1, 'padding':1, 'act':'relu', 'bias_attr':True},
                {'type':Conv2D, 'num_channels':256, 'num_filters':256, 'filter_size':3, 'stride':1, 'padding':1, 'act':'relu', 'bias_attr':True},
                {'type':Conv2D, 'num_channels':256, 'num_filters':256, 'filter_size':3, 'stride':1, 'padding':1, 'act':'relu', 'bias_attr':True},

                {'type':Pool2D, 'pool_size':2,     'pool_type':'max',    'pool_stride':2,         'global_pooling':False},

                {'type':Conv2D, 'num_channels':256, 'num_filters':512, 'filter_size':3, 'stride':1, 'padding':1, 'act':'relu', 'bias_attr':True},
                {'type':Conv2D, 'num_channels':512, 'num_filters':512, 'filter_size':3, 'stride':1, 'padding':1, 'act':'relu', 'bias_attr':True},
                {'type':Conv2D, 'num_channels':512, 'num_filters':512, 'filter_size':3, 'stride':1, 'padding':1, 'act':'relu', 'bias_attr':True},

                {'type':Pool2D, 'pool_size':2,     'pool_type':'max',    'pool_stride':2,         'global_pooling':False},

                {'type':Conv2D, 'num_channels':512, 'num_filters':512, 'filter_size':3, 'stride':1, 'padding':1, 'act':'relu', 'bias_attr':True},
                {'type':Conv2D, 'num_channels':512, 'num_filters':512, 'filter_size':3, 'stride':1, 'padding':1, 'act':'relu', 'bias_attr':True},
                {'type':Conv2D, 'num_channels':512, 'num_filters':512, 'filter_size':3, 'stride':1, 'padding':1, 'act':'relu', 'bias_attr':True},

                {'type':Pool2D, 'pool_size':2,     'pool_type':'max',    'pool_stride':2,         'global_pooling':False},

            ]
        else:
            self.VGG_part_list1 = copy.deepcopy(VGG_part_list1)


        if VGG_part_list2 is None:

            self.VGG_part_list2 = [

                {'type':Linear, 'input_dim': 512*7*7, 'output_dim':4096,      'act':'relu', 'bias_attr':True},
                {'type':Linear, 'input_dim': 4096,    'output_dim':4096,      'act':'relu', 'bias_attr':True},
                {'type':Linear, 'input_dim': 4096,    'output_dim':out_dim,   'act':'relu', 'bias_attr':True},
            ]
        else:
            self.VGG_part_list2 = copy.deepcopy(VGG_part_list2)


        self.VGG_part1 = LinConPoo(self.VGG_part_list1)
        self.VGG_part2 = LinConPoo(self.VGG_part_list2)



    def forward(self, inputs):

    	# VGG的第一部分, `flatten`的前半部分
        VGG_part1 = self.VGG_part1(inputs)
        x = fluid.layers.flatten(VGG_part1)
        # VGG的第二部分, `flatten`的后半部分
        VGG_part2 = self.VGG_part2(x)

        return VGG_part2

class Inception_v1(fluid.dygraph.Layer):

    def __init__(self, num_channels, ch1x1, ch3x3reduced, ch3x3, ch5x5reduced, ch5x5, pool_proj):
        '''
        @Brief
            传入参数用于定义 `Inception_v1` 结构

        @Parameters
            num_channels : 传入图片通道数
            ch1x1        : 1x1卷积操作的输出通道数
            ch3x3reduced : 3x3卷积之前的1x1卷积的通道数
            ch3x3        : 3x3卷积操作的输出通道数
            ch5x5reduced : 5x5卷积之前的1x1卷积的通道数
            ch5x5        : 5x5卷积操作的输出通道数
            pool_proj    : 池化操作之后1x1卷积的通道数

        @Return
            返回 `Inception_v1` 网络模型

        @Examples
        ------------
        '''

        super(Inception_v1, self).__init__()

        self.branch1 = Conv2D(num_channels=num_channels,
                              num_filters=ch1x1,
                              filter_size=1,
                              stride=1,
                              act='relu',
                              padding=0)


        branch2_list = [
                {'type':Conv2D, 'num_channels': num_channels, 'num_filters':ch3x3reduced, 'filter_size':1, 'stride':1, 'padding':0, 'act':'relu', 'bias_attr':True},
                {'type':Conv2D, 'num_channels':ch3x3reduced,  'num_filters':ch3x3,        'filter_size':3, 'stride':1, 'padding':1, 'act':'relu', 'bias_attr':True},
        ]
        self.branch2 = LinConPoo(branch2_list)

        branch3_list = [
                {'type':Conv2D, 'num_channels': num_channels, 'num_filters':ch5x5reduced, 'filter_size':1, 'stride':1, 'padding':0, 'act':'relu', 'bias_attr':True},
                {'type':Conv2D, 'num_channels':ch5x5reduced,  'num_filters':ch5x5,        'filter_size':5, 'stride':1, 'padding':2, 'act':'relu', 'bias_attr':True},
        ]
        self.branch3 = LinConPoo(branch3_list)

        branch4_list = [
                {'type':Pool2D,  'pool_size':3,  'pool_type':'max',  'pool_stride':1,  'pool_padding':2,  'global_pooling':False},
                {'type':Conv2D,  'num_channels':num_channels, 'num_filters':pool_proj, 'filter_size':5, 'stride':1, 'padding':1, 'act':'relu', 'bias_attr':True},
        ]
        self.branch4 = LinConPoo(branch4_list)



    def forward(self, inputs):
        '''
        @Parameters :
            inputs     :   原始数据
        '''

        branch1 = self.branch1(inputs)
        branch2 = self.branch2(inputs)
        branch3 = self.branch3(inputs)
        branch4 = self.branch4(inputs)

        outputs = concat([branch1, branch2, branch3, branch4], axis=1)

        return outputs


class GoogLeNet(fluid.dygraph.Layer):

    def __init__(self, num_channels=3, out_dim=2):
        '''
        @Brief:
            使用 `Inception_v1` 结构搭建的 `GoogLeNet` 模型
            注: 喂入的图片最好是 224 * 224
        @Parameters:
            num_channels : 输入的图片通道数
            out_dim      : 输出的维度(几分类就是几)
        @Return:
            out          : 主输出(shape=(X, out_dim))
            out1         : 辅助分类器_1的输出(shape=(X, out_dim))
            out2         : 辅助分类器_2的输出(shape=(X, out_dim))
        @Examples:
        ------------
        >>> import numpy as np
        >>> data = np.ones(shape=(8, 3, 224, 224), dtype=np.float32) # 假设为8张三通道的照片
        >>> with fluid.dygraph.guard():
                googlenet = GoogLeNet(out_dim=10)
                data = fluid.dygraph.to_variable(data)
                y, _, _ = googlenet(data)
                print(y.numpy().shape)
        (8, 10)
        '''

        super(GoogLeNet, self).__init__()

        part1_list  = [
            {'type':Conv2D, 'num_channels':num_channels, 'num_filters':64, 'filter_size':7, 'stride':2, 'padding':3, 'act':None, 'bias_attr':False},
            {'type':Pool2D, 'pool_size':3, 'pool_type':'max', 'pool_stride':2, 'pool_padding':0, 'global_pooling':False},
        ]

        part2_list  = [
            {'type':Conv2D, 'num_channels':64, 'num_filters':64 , 'filter_size':1, 'stride':1, 'padding':0, 'act':None, 'bias_attr':False},
            {'type':Conv2D, 'num_channels':64, 'num_filters':192, 'filter_size':3, 'stride':1, 'padding':1, 'act':None, 'bias_attr':False},
        ]



        self.googLeNet_part1 = Sequential(
                                ('part1', LinConPoo(part1_list)),
                                ('BN1', BatchNorm(64)),
                                ('part2', LinConPoo(part2_list)),
                                ('BN2', BatchNorm(192)),
                                ('MaxPool1', Pool2D(pool_size=3, pool_type='max', pool_stride=2)),
                                ('inception_3a', Inception_v1(192,  64,  96, 128, 16, 32, 32)),
                                ('inception_3b', Inception_v1(256, 128, 128, 192, 32, 96, 64)),
                                ('MaxPool2', Pool2D(pool_size=3, pool_type='max', pool_stride=2)),
                                ('inception_4a', Inception_v1(480, 192, 96, 208, 16, 48, 64)),
                            )

        # `self.googLeNet_part1` 完成了 `inception_4a` 之前的部分, 此处需要辅助分类器
        self.auxiliary_classifier1_1 = LinConPoo([
            {'type':Pool2D, 'pool_size':5, 'pool_type':'avg', 'pool_stride':3, 'pool_padding':0, 'global_pooling':False},
            {'type':Conv2D, 'num_channels':512, 'num_filters':128, 'filter_size':1, 'stride':1, 'padding':0, 'act':None, 'bias_attr':False},
        ])
        self.auxiliary_classifier1_fc1 = Linear(input_dim=128*3*3, output_dim=1024, act='relu', bias_attr=True)
        self.auxiliary_classifier1_fc2 = Linear(input_dim=1024, output_dim=out_dim, act='softmax', bias_attr=True)



        # 此处开始定义辅助分类器之后的部分
        self.googLeNet_part2 = Sequential(
                                # ('googLeNet_part1', self.googLeNet_part1),
                                ('inception_4b', Inception_v1(512, 160, 112, 224, 24, 64, 64)),
                                ('inception_4c', Inception_v1(512, 128, 128, 256, 24, 64, 64)),
                                ('inception_4d', Inception_v1(512, 112, 144, 288, 32, 64, 64)),
                            )

        # `self.googLeNet_part2`完成了 `inception_4e` 之前的部分, 此处需要辅助分类器
        self.auxiliary_classifier2_1 = LinConPoo([
            {'type':Pool2D, 'pool_size':5, 'pool_type':'avg', 'pool_stride':3, 'pool_padding':0, 'global_pooling':False},
            {'type':Conv2D, 'num_channels':512, 'num_filters':128, 'filter_size':1, 'stride':1, 'padding':0, 'act':None, 'bias_attr':False},
        ])
        self.auxiliary_classifier2_fc1 = Linear(input_dim=128*3*3, output_dim=1024, act='relu', bias_attr=True)
        self.auxiliary_classifier2_fc2 = Linear(input_dim=1024, output_dim=out_dim, act='softmax', bias_attr=True)



        # 此处开始定义辅助分类器之后的部分
        self.googLeNet_part3 = Sequential(
                                # ('googLeNet_part2', self.googLeNet_part2),
                                ('inception_4e', Inception_v1(528, 256, 160, 320, 32, 128, 128)),
                                ('MaxPool3', Pool2D(pool_size=3, pool_type='max', pool_stride=2)),
                                ('inception_5a', Inception_v1(832, 256, 160, 320, 32, 128, 128)),
                                ('inception_5b', Inception_v1(832, 384, 192, 384, 48, 128, 128)),
                                ('AvgPool1', Pool2D(pool_size=6, pool_type='max', pool_stride=1)),
                            )
        # 由于 `Sequential` 中不能添加 dropout 层, 所以此处仍然要分割
        self.last_fc = Linear(1024, out_dim, act='softmax', bias_attr=True)


    def forward(self, inputs):

        # 没有辅助分类器之前的部分
        googLeNet_part1 = self.googLeNet_part1(inputs)
        # ------------------- 由辅助分类器来分隔 -------------------
        googLeNet_part2 = self.googLeNet_part2(googLeNet_part1)
        # ------------------- 由辅助分类器来分隔 -------------------
        googLeNet_part3 = self.googLeNet_part3(googLeNet_part2)
        # 将输出拉直
        out = fluid.layers.flatten(googLeNet_part3, axis=1)
        out = fluid.layers.dropout(x=out, dropout_prob=0.7)
        out = self.last_fc(out)


        # 第一个辅助分类器
        out1 = self.auxiliary_classifier1_1(googLeNet_part1)
        out1 = fluid.layers.flatten(out1, axis=1)
        out1 = self.auxiliary_classifier1_fc1(out1)
        out1 = fluid.layers.dropout(x=out1, dropout_prob=0.7)
        out1 = self.auxiliary_classifier1_fc2(out1)


        # 第二个辅助分类器
        out2 = self.auxiliary_classifier2_1(googLeNet_part1)
        out2 = fluid.layers.flatten(out2, axis=1)
        out2 = self.auxiliary_classifier2_fc1(out2)
        out2 = fluid.layers.dropout(x=out2, dropout_prob=0.7)
        out2 = self.auxiliary_classifier2_fc2(out2)

        return out, out1, out2
