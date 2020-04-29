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
