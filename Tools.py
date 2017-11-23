import os
import time
import tensorflow as tf
from collections import OrderedDict


class Tools:
    def __init__(self):
        pass

    @staticmethod
    def print_info(info):
        print(time.strftime("%H:%M:%S", time.localtime()), info)
        pass

    # 新建目录
    @staticmethod
    def new_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    # 收集可训练变量
    @staticmethod
    def collect_vars(scope, start=None, end=None, prepend_scope=None):
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        var_dict = OrderedDict()
        if isinstance(start, str):
            for i, var in enumerate(vars):
                var_name = '/'.join(var.op.name.split('/')[1:])
                if var_name.startswith(start):
                    start = i
                    break
                pass
        if isinstance(end, str):
            for i, var in enumerate(vars):
                var_name = '/'.join(var.op.name.split('/')[1:])
                if var_name.startswith(end):
                    end = i
                    break
                pass
        for var in vars[start:end]:
            var_name = '/'.join(var.op.name.split('/')[1:])
            if prepend_scope is not None:
                var_name = os.path.join(prepend_scope, var_name)
            var_dict[var_name] = var
        return var_dict

    pass