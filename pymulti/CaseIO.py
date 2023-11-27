# 导入库的方法：import pymulti.CaseIO
from . import os
from . import shutil
from . import subprocess
from . import re
from . import warnings
from . import Multi_Program
import struct


def merge_feature(list1: list, list2: list):
    """
    合并特征字符串和代写入的值为replace_list格式，注意此次应当输入两个一维数组
    """
    if len(list1) == len(list2):
        return [[list1[i], list2[i]] for i in range(len(list1))]
    raise ValueError('list1和list2的长度应当相等')


class Cases():
    def __init__(self, program: str, CaseDir: str, source_path: str, target_path: str, replace_list=None, file_path='/User.r'):
        """
        feature如果多次出现,开启rep_virable后将会全部替换
        否则可以使用$符号来区分不同的feature
        如对于同一个feature可以使用'feature$0','feature$1'来区分,
        并且如果多于或者少于实际出现的次数，将会按照较少的次数进行替换
        """
        self.program = program  # 程序
        self.CaseDir = CaseDir  # 运行cases文件夹
        self.source_path = self.CaseDir+source_path  # 复制父本路径
        self.target_path = self.CaseDir+target_path  # 复制子本路径
        self.file_path = self.target_path+file_path  # 修改文件路径，通常为默认值
        # 需要修改的值，二维列表，如[[feature,new_val],...][]
        self.replace_list = replace_list
        # 需要修改的值的行号，二维列表，如[[feature,line1,line2],...]
        self.feature_num_list = {}
        self.content = None  # 文件内容

    def __mkdir_(self):
        if not os.path.exists(self.target_path):
            # 如果目标路径不存在原文件夹的话就创建
            os.makedirs(self.target_path)
        if os.path.exists(self.target_path):
            # 如果目标路径存在原文件夹的话就先删除
            warnings.warn(f'{self.target_path} 目标路径已存在，将删除该路径下所有文件')
            shutil.rmtree(self.target_path)
            # 如果目标路径存在，跳过下面操作，进入下一次循环
            # continue
        shutil.copytree(self.source_path, self.target_path)

    def __precheck_(self):
        for r_list in self.replace_list:
            if not (type(r_list) == list and len(r_list) == 2):
                raise ValueError(
                    'replace_list应当是二维列表，如[[feature,new_val],...]')

    def __get_feature_name_(self):
        for rl in self.self.replace_list:
            temp_f = rl[0]
            feature = temp_f if temp_f.find(
                '$') == -1 else temp_f[:temp_f.find('$')]
            if not feature in self.feature_num_list:
                self.feature_num_list[feature] = []
        self.feature_num_list = [[fl] for fl in self.feature_num_list]

    def __get_config_(self):
        # 读取文件内容
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.content = f.readlines()
        for i, line in enumerate(self.content):
            for fl in self.feature_num_list:
                if re.findall(f'{str(fl[0])}.*?=.*?;', line):
                    self.feature_num_list[fl].append(i)
                    break

    def __write_config_(self):
        # 写入文件内容
        with open(self.file_path, "w", encoding="utf-8") as f:
            f.writelines(self.content)

    def __update_feature_(self, feature: str, new_val: str):
        """
        替换文件中的字符串
        :param file:文件名
        :param feature:需要替换的变量名
        :param new_val:新的变量值,必须为字符串
        """
        feature = feature if feature.find(
            '$') == -1 else feature[:feature.find('$')]
        do_replace = False if feature.find('$') == -1 else True
        if do_replace:
            index = 0 if feature.find(
                '$') == -1 else int(feature[feature.find('$')+1:])
            list = self.feature_num_list[feature]
            if len(list) <= index:
                line_num = list[index]
                self.content[line_num] = re.sub(
                    f'{str(feature)}.*?=.*?;', f"{str(feature)} = {str(new_val)};", self.content[line_num])
        else:
            for line_num in self.feature_num_list[feature]:
                self.content[line_num] = re.sub(
                    f'{str(feature)}.*?=.*?;', f"{str(feature)} = {str(new_val)};", self.content[line_num])

    def new_case(self):
        # 生成新case
        self.__precheck_()
        self.__mkdir_()
        self.__get_feature_name_()
        self.__get_config_()
        for r_list in self.replace_list:
            feature, new_val = r_list
            self.__update_feature_(feature, new_val)
        self.__write_config_()

    def run(self):
        """
        运行该case
        """
        runorder = 'cd '+self.target_path
        p = subprocess.run(runorder+' && ./RUN', shell=True)
        return p  # 返回进程对象，用于等待进程结束

    def __data_Struct_(self, filename):
        # here filename should be ***.d
        with open(filename, 'r') as file:
            dataStr = file.read()
            dataLst = dataStr.split()
            dataLst = dataLst[1:]  # remove the first useless item

            dataTable = [[dataLst[i], int(dataLst[i+1]), int(dataLst[i+2])]
                         for i in range(0, len(dataLst), 3)]
            # resize the list to 3xN and convert strings into integers
            # print(dataTable)
            return dataTable

    def __data_Input_(self, filename):
        # return all data from ***, here filename = *** should be a number (time)
        dataLst = []
        with open(filename, 'rb') as dataSet:
            float_bytes = dataSet.read(4)
            while float_bytes:
                float_value = struct.unpack('f', float_bytes)[0]
                # print(float_value)
                float_bytes = dataSet.read(4)
                dataLst.append(float_value)
        return dataLst[1:]

    def data_tag_get(self, filename, tag):
        # get part of data in ***, here filename = *** should be a number (time), and tag should be a string (eg "cn")
        if self.program == Multi_Program.multi_1d:
            pass
        elif self.program == Multi_Program.multi_2d:
            filename = self.target_path+"/"+filename
        elif self.program == Multi_Program.multi_3d:
            filename = self.target_path+"_3DM/"+filename
        dataStructure = self.__data_Struct_(filename+".d")
        dataSet = self.__data_Input_(filename)
        for item in dataStructure:
            if item[0] == tag:
                dataSelected = dataSet[item[1]-1:item[1]-1+item[2]]
        return dataSelected
