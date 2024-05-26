# 导入库的方法：import pymulti.CaseIO
import os
import subprocess
import re
import warnings
from .utils import Multi_Program
import struct
import numpy as np
import glob
import matplotlib.pyplot as plt


def merge_feature(list1: list, list2: list):
    """
    合并特征字符串和代写入的值为replace_list格式，注意此次应当输入两个一维数组
    """
    if len(list1) == len(list2):
        return [[list1[i], list2[i]] for i in range(len(list1))]
    raise ValueError('list1和list2的长度应当相等')


class Cases():
    def __init__(self, program: str, CaseDir: str, source_path: str, target_path: str, replace_list: list = None, file_path: str = '/User.r'):
        """
        program:
            Multi_Program类型，指定要初始化的程序
        CaseDir: 
            str类型，指定case所在的文件夹
        source_path: 
            str类型，指定源文件路径（相对于CaseDir）
        target_path: 
            str类型，指定目标文件路径（相对于CaseDir）
        replace_list: 
            list类型，指定要替换的参数列表，格式为[[feature,new_val],...]
            如对于同一个feature可以使用'feature$0','feature$1'来区分，
            如果直接使用'feature'则会替换所有该feature出现的地方，
            并且如果超出实际该feature出现的次数时会自动忽略
        file_path: 
            str类型，指定要修改的文件路径，默认为/User.r
        """
        self.program = program  # 程序
        self.CaseDir = CaseDir  # 运行cases文件夹
        self.source_path = self.CaseDir+source_path  # 复制父本路径
        self.target_path = self.CaseDir+target_path  # 复制子本路径
        self.file_path = self.target_path+file_path  # 修改文件路径，通常为默认值
        # 需要修改的值，二维列表，如[[feature$0,new_val],...]
        self.replace_list = replace_list
        self.feature_num_list = {}
        self.content = None  # 文件内容

    def __mkdir_(self):
        '''
        创建目标文件夹
        '''
        cp = f'cd {self.CaseDir};mkdir {self.target_path} 2> /dev/null;cd {self.source_path};cp -p $(cat FILELIST) /{self.target_path}'
        os.system(cp)

    def __precheck_(self):
        '''
        预检查替换列表是否符合规范
        '''
        for r_list in self.replace_list:
            if not (type(r_list) == list and len(r_list) == 2):
                raise ValueError(
                    'replace_list应当是二维列表，如[[feature,new_val],...]')

    def __get_feature_name_(self):
        '''
        获取需要替换的特征名
        '''
        for rl in self.replace_list:
            temp_f = rl[0]
            feature = temp_f if temp_f.find(
                '$') == -1 else temp_f[:temp_f.find('$')]
            if not feature in self.feature_num_list:
                self.feature_num_list[feature] = []

    def __get_config_(self):
        '''
        读取文件内容，获取需要的特征所在的所有行数
        '''
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.content = f.readlines()
        for i, line in enumerate(self.content):
            for fl in self.feature_num_list:
                if re.findall(f'{fl}\s+=.*?;', line):
                    self.feature_num_list[fl].append(i)
                    break

    def __write_config_(self):
        '''
        写入文件内容
        '''
        with open(self.file_path, "w", encoding="utf-8") as f:
            f.writelines(self.content)

    def __update_feature_(self, feature: str, new_val: str):
        """
        替换文件中的字符串

        feature:
            需要替换的变量名
        new_val:
            新的变量值,必须为字符串
        """
        do_replace = feature.find('$') != -1
        if do_replace:
            ind = feature.find('$')
            index = int(feature[ind+1:])
            feature = feature[:feature.find('$')]
            list = self.feature_num_list[feature]
            if len(list) > index:
                line_num = list[index]
                self.content[line_num] = re.sub(
                    f'{feature}\s+=.*?;', f"{feature} = {new_val};", self.content[line_num])
        else:
            for line_num in self.feature_num_list[feature]:
                self.content[line_num] = re.sub(
                    f'{feature}\s+=.*?;', f"{feature} = {new_val};", self.content[line_num])

    def get_origin_param(self, feature: str) -> list:
        '''
        从user.r获取原始参数
        feature: 
            str类型，指定要查找的特征名
        '''
        content_find = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.content = f.readlines()
        for line in self.content:
            ans = re.findall(f'{feature}\s+=.*?;', line)
            # print(ans)
            if ans:
                content_find.append(ans[0])
        return content_find

    def new_case(self):
        '''
        生成新case
        '''
        self.__mkdir_()
        print('dir made')
        if self.replace_list == None:
            warnings.warn('replace_list为空，将不会进行替换')
            return
        self.__precheck_()
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
        print(runorder)
        subprocess.run(runorder+';pwd;'+' ./RUN', shell=True)

    def __data_Struct_(self, filename):
        # here filename should be ***.d
        with open(filename, 'r') as file:
            dataStr = file.read()
            dataLst = dataStr.split()
            dataLst = dataLst[1:]  # remove the first useless item

            dataTable = [[dataLst[i], int(dataLst[i+1]), int(dataLst[i+2])]
                         for i in range(0, len(dataLst), 3)]
            # resize the list to 3xN and convert strings into integers
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

    def get_data_tag(self, filename, tag):
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

    def __tri2node_(self, tn, Tn):
        # tri2node function
        # @author: kglize
        # Tn: triangle node table
        # tn: table of triangle-node connectivity
        nt = tn.shape[0]  # number of triangles
        npx = max(max(tn[:, 0]), max(tn[:, 1]),
                  max(tn[:, 2]))  # number of nodes
        # count the number of triangles that a node belongs to
        count = np.zeros(npx)
        # store the number of triangles that a node belongs to
        Quant = np.zeros(npx)

        # loop on all triangles to accumulate counts and quantities of a node
        for i in range(nt):
            for j in range(3):
                n = tn[i, j]
                count[n - 1] += 1
                Quant[n - 1] += Tn[i]

        return Quant / count

    def __node2tri_(self, tn, Tn):
        # @author: kglize
        # Tn: triangle node table
        # tn: table of triangle-node connectivity
        nt = tn.shape[0]  # number of triangles
        # store the number of triangles that a node belongs to
        Tc = np.zeros(nt, dtype=np.int32)

        # loop on all triangles to accumulate counts and quantities of a node
        for i in range(nt):
            Tc[i] = (Tn[tn[i, 0] - 1] + Tn[tn[i, 1] - 1] + Tn[tn[i, 2] - 1]) / 3

        return Tc

    def getTimeTable(self):
        # @author: kglize
        if self.program == Multi_Program.multi_1d:
            directory = self.target_path
        elif self.program == Multi_Program.multi_2d:
            directory = self.target_path
        elif self.program == Multi_Program.multi_3d:
            directory = self.target_path+"_3DM"
        # get time table in the directory
        files = glob.glob(directory + '/*.d')
        filenames = [os.path.basename(x) for x in files]
        stime = np.zeros(len(filenames))
        for i in range(len(filenames)):
            a = filenames[i]
            length = len(a)
            stime[i] = float(a[0:length - 2])
        time = np.sort(stime)
        return time

    def get_tn(self):
        if self.program == Multi_Program.multi_1d:
            directory = self.target_path
        elif self.program == Multi_Program.multi_2d:
            directory = self.target_path
        elif self.program == Multi_Program.multi_3d:
            directory = self.target_path+"_3DM"

        fp01 = open(directory + '/0.d', 'r')
        mf01 = np.loadtxt(fp01, dtype={'names': (
            'name', 'start', 'length'), 'formats': ('U10', 'i4', 'i4')}, skiprows=1)
        spt = mf01['start'][0]
        lpt = mf01['length'][0]
        sct = mf01['start'][1]
        lct = mf01['length'][1]
        # open file MF02 to get data
        fp02 = open(directory + '/0', 'rb')
        mf02 = np.fromfile(fp02, dtype=np.float32)
        mf02 = np.delete(mf02, 0)  # delete first element, i.e, MF02
        pt = mf02[spt:spt + lpt]  # table of triangle-node connectivity
        # arrange the connectivity of triangles in the required matrix form
        nt = lct  # number of triangles
        tn = np.zeros((nt, 3), dtype=np.int32)  # connectivity table
        for i in range(nt):
            tn[i, 0] = pt[3 * i]
            tn[i, 1] = pt[3 * i + 1]
            tn[i, 2] = pt[3 * i + 2]
        return nt, tn

    def get_data_T(self, time, tn, nt, directory):
        """
        get the data at time[i] and rearrange the data into the triangle form
        """
        # open description file to get data structure of file MF02
        fp1 = open(directory + '/' + f'{time:g}' + '.d', 'r')
        mf1 = np.loadtxt(fp1, dtype={'names': (
            'name', 'start', 'length'), 'formats': ('U10', 'i4', 'i4')}, skiprows=1)
        # mf1 = np.genfromtxt(fp1, dtype = ['U10', 'i4', 'i4'], names = ('name', 'start', 'length'), skip_header=1)

        for j in range(len(mf1)):
            if mf1['name'][j] == 'x':
                sx = mf1['start'][j]
                lx = mf1['length'][j]
            elif mf1['name'][j] == 'y':
                sy = mf1['start'][j]
                ly = mf1['length'][j]
            elif mf1['name'][j] == 'rho':
                srho = mf1['start'][j]
                lrho = mf1['length'][j]
            elif mf1['name'][j] == 'P':
                sP = mf1['start'][j]
                lP = mf1['length'][j]
            elif mf1['name'][j] == 'T':
                sT = mf1['start'][j]
                lT = mf1['length'][j]
            elif mf1['name'][j] == 'TR':
                sTR = mf1['start'][j]
                lTR = mf1['length'][j]
            elif mf1['name'][j] == 'frac1':
                sfrac1 = mf1['start'][j]
                lfrac1 = mf1['length'][j]
            elif mf1['name'][j] == 'xraypower':
                sPxray = mf1['start'][j]
                lPxray = mf1['length'][j]

        # off set start point when time>0
        istart = (time > 0) * 4 * nt * 0

        # open file MF02 to get data
        fp2 = open(directory + '/' + f'{time:g}', 'rb')
        mf2 = np.fromfile(fp2, dtype=np.float32)
        mf2 = np.delete(mf2, 0)  # delete first element, i.e, MF02

        x = mf2[sx - istart - 1:sx - istart + lx - 1] * 1e4  # x coordinate
        y = mf2[sy - istart - 1:sy - istart + ly - 1] * 1e4  # y coordinate
        rho = mf2[srho - istart - 1:srho - istart + lrho - 1]  # density
        P = mf2[sP - istart - 1:sP - istart + lP - 1]  # pressure
        T = mf2[sT - istart - 1:sT - istart + lT - 1]  # temperature
        frac1 = mf2[sfrac1 - istart - 1:sfrac1 -
                    istart + lfrac1 - 1]  # fraction of phase 1

        # rearrange data
        rn = np.sqrt(x ** 2 + y ** 2)
        frac1n = self.__tri2node_(tn, frac1)
        rhoDT = rho * frac1
        Tc = self.__node2tri_(tn, T)
        Pc = self.__node2tri_(tn, P)
        xc = self.__node2tri_(tn, x)
        yc = self.__node2tri_(tn, y)
        return rn, frac1n, rhoDT, Tc, Pc, xc, yc, x, y, rho, P, T, frac1

    def getData(self):
        # open description file to get data structure of file MF02
        # @author: kglize

        if self.program == Multi_Program.multi_1d:
            directory = self.target_path
        elif self.program == Multi_Program.multi_2d:
            directory = self.target_path
        elif self.program == Multi_Program.multi_3d:
            directory = self.target_path+"_3DM"

        nt, tn = self.get_tn()  # get triangle-node connectivity

        # get data at different time
        time = self.getTimeTable()
        nTime = len(time)

        rhomax = np.zeros(nTime)
        Pmax = np.zeros(nTime)
        Tmax = np.zeros(nTime)
        for i in range(nTime):
            rn, frac1n, rhoDT, Tc, Pc, xc, yc, x, y, rho, P, T, frac1 = self.get_data_T(
                time[i], tn, nt, directory)
            # find the max density and the corresponding temperature
            rhoind = 0.98 * np.max(rhoDT * (xc > yc))
            mindex = np.where(rhoDT * (xc > yc) > rhoind)[0][0]
            rhomax[i] = rhoDT[mindex]
            Tmax[i] = Tc[mindex]
            Pmax[i] = P[mindex]
            print(i)
        return rhomax, Tmax, Pmax, time

    def plotRhoTP(self):
        # plotRhoTP function
        # @author: kglize
        # set end time
        if self.program == Multi_Program.multi_1d:
            directory = self.target_path
        elif self.program == Multi_Program.multi_2d:
            directory = self.target_path
        elif self.program == Multi_Program.multi_3d:
            directory = self.target_path+"_3DM"

        endTime = 0.6e-9
        rhomax, Tmax, Pmax, time = self.getData(directory)  # get data

        # plot
        fig, ax = plt.subplots(figsize=(12, 6))
        mend = np.where(time > endTime)[0][0]
        TFermi = 14.05 * rhomax ** (2 / 3)
        ax.plot(rhomax[:mend], Tmax[:mend], 'k-', label='Temperature')
        ax.plot(rhomax[:mend], TFermi[:mend], 'r-', label='Fermi Temperature')
        ax.set_xlabel(r'Density(g/cc)')
        ax.set_ylabel(r'T(eV)')
        ax.legend()
        plt.show()

    def get_coordinate(self, filename):
        x_list = self.get_data_tag(filename, "x")
        y_list = self.get_data_tag(filename, "y")
        z_list = self.get_data_tag(filename, "z")
        return x_list, y_list, z_list
