from . import os
from . import shutil
from . import subprocess
from . import re
from . import warnings

def merge_feature(list1:list,list2:list):
    """
    合并特征字符串和代写入的值为replace_list格式，注意此次应当输入两个一维数组
    """
    if len(list1)==len(list2):
        return [[list1[i],list2[i]] for i in range(len(list1))]
    raise ValueError('list1和list2的长度应当相等')

class Cases():
    def __init__(self,CaseDir:str,source_path:str,target_path:str,replace_list=None,file_path='/User.r'):
        self.CaseDir=CaseDir#运行cases文件夹
        self.source_path=source_path#复制父本路径
        self.target_path=target_path#复制子本路径
        self.file_path=file_path#修改文件路径，通常为默认值
        self.replace_list=replace_list#需要修改的值，二维列表，如[[feature,new_val],...]

    def __mkdir_(self):
        if not os.path.exists(self.target_path):
            # 如果目标路径不存在原文件夹的话就创建
            os.makedirs(self.target_path)
        if os.path.exists(self.target_path):
            # 如果目标路径存在原文件夹的话就先删除
            warnings.warn(f'{self.target_path} 目标路径已存在，将删除该路径下所有文件')
            shutil.rmtree(self.target_path)
            # 如果目标路径存在，跳过下面操作，进入下一次循环
            #continue
        shutil.copytree(self.source_path, self.target_path)

    def __update_feature_(self,feature:str,new_val:str):
        """
        替换文件中的字符串
        :param file:文件名
        :param feature:需要替换的变量名
        :param new_val:新的变量值,必须为字符串
        """
        file=self.file_path
        with open(file, "r", encoding="utf-8") as f:
            content=f.readlines()
            replaced_content=re.sub(f'{str(feature)}.*?=.*?;',f"{str(feature)} = {str(new_val)};",content)
        with open(file,"w",encoding="utf-8") as f:
            f.write(replaced_content)
        if content==replaced_content:
            warnings.warn(f'未找到{str(feature)},请检查{file}')

    def new_case(self):
        #生成新case
        self.__mkdir_()
        for r_list in self.replace_list:
            if type(r_list)==list and len(r_list)==2:
                feature,new_val=r_list
                self.__update_feature_(feature,new_val)
            else:
                raise ValueError('replace_list应当是二维列表，如[[feature,new_val],...]')

    def run(self):
        """
        运行该case
        """
        runorder = 'cd '+self.target_path
        p=subprocess.run(runorder+' && ./RUN', shell = True)
        return p#返回进程对象，用于等待进程结束

    def data_get(self):
        """
        读取数据
        """
        pass