from .CaseIO import Cases
import scipy.interpolate
import numpy as np
from .utils import Multi_Program, print
import matplotlib.pyplot as plt

## 全局变量
RM_ANGLE=40.0 #边缘仰角，大于这个角度的样本将不被考虑，设置为金锥内壁的角度
THETA_DIV=41 # 角方向的切分段数（矩形插值）
ZENITH_DIV=41 # 天顶角方向的切分段数
AZIMUTH_DIV=181 # 方位角方向的切分段数
R_MIN_INTERVAL=0.001 # 径向最小被划分区间长度
R_DIV_INTERVAL=5 # 径向每个被划分区间被划分出的段数
RHO_INITIAL_VALUE=1 # 未被压缩区域的密度
RHO_EFFECT_RATIO=1.1 # 密度大于 RHO_EFFECT_RATIO*RHO_INITIAL_VALUE 的区域被考虑
RHO_EFFECT_RATIO_FINE=1.2 # 密度大于 RHO_EFFECT_RATIO_FINE*RHO_INITIAL_VALUE 的区域被计算入积分
SHOCK_WAVE_DIV=100 # 激波区的切分段数

THETA_LIST=[-RM_ANGLE+i*2*RM_ANGLE/(THETA_DIV-1) for i in range(THETA_DIV)] # 角方向差值点
THETA_CENTER=int((THETA_DIV-1)/2); # 中央射线的序号
ZENITH_LIST=[i*RM_ANGLE/(ZENITH_DIV-1) for i in range(ZENITH_DIV)] # 天顶角方向差值点
AZIMUTH_LIST=[i*360.0/(AZIMUTH_DIV-1) for i in range(AZIMUTH_DIV)] # 天顶角方向差值点

def write_float_array_to_txt(float_array, filename):
    """
    将浮点数组写入txt文件
    
    参数:
        float_array: 浮点数组
        filename: 要写入的文件名
    """
    # 打开一个txt文件以写入模式
    with open(filename, "w") as file:
        # 使用循环遍历数组，并将每个元素写入文件
        for num in float_array:
            if np.isnan(num):
                file.write("nan\n")
            else:
                file.write("%.6f\n" % num)  # 将浮点数格式化为字符串，并写入文件
                
def write_float_matrix_to_txt(float_matrix, filename):
    """
    将浮点数组写入txt文件
    
    参数:
        float_matrix: 浮点矩阵
        filename: 要写入的文件名
    """
    # 打开一个txt文件以写入模式
    with open(filename, "w") as file:
        # 使用循环遍历数组，并将每个元素写入文件
        for i in range(len(float_matrix)):
            for j in range(len(float_matrix[0])):
                if np.isnan(float_matrix[i][j]):
                    file.write("nan ")
                else:
                    file.write("%.6f " % float_matrix[i][j])  # 将浮点数格式化为字符串，并写入文件
            file.write("\n")

def rho_cell2node(cn_list, rho_list):
    # convert the list of rho into a list with the same length as x list
    nt = len(cn_list)//4
    cn_list_reform = [[int(cn_list[i])-1, int(cn_list[i+1])-1, int(cn_list[i+2])-1,
                       int(cn_list[i+3])-1] for i in range(0, len(cn_list), 4)]
    nnd = int(max(cn_list))
    # print(nt,np)
    count = [0]*nnd
    Quant = [0]*nnd

    for i in range(nt):
        count[cn_list_reform[i][1]] = count[cn_list_reform[i][1]]+1
        Quant[cn_list_reform[i][1]] = Quant[cn_list_reform[i][1]]+rho_list[i]

        count[cn_list_reform[i][2]] = count[cn_list_reform[i][2]]+1
        Quant[cn_list_reform[i][2]] = Quant[cn_list_reform[i][2]]+rho_list[i]

        count[cn_list_reform[i][3]] = count[cn_list_reform[i][3]]+1
        Quant[cn_list_reform[i][3]] = Quant[cn_list_reform[i][3]]+rho_list[i]

        count[cn_list_reform[i][0]] = count[cn_list_reform[i][0]]+1
        Quant[cn_list_reform[i][0]] = Quant[cn_list_reform[i][0]]+rho_list[i]

    return [x/y for x, y in zip(Quant, count)]

def gen_polar_coor_grid__cubic(x_list, y_list, z_list):
    # 方形的一个极坐标网格点阵 
    nnd = len(x_list)
    r_list = [np.sqrt(x_list[i]**2+y_list[i]**2+z_list[i]**2)
              for i in range(nnd)]

    sorted_r_list=sorted(r_list)
    reduced_r_list=[]
    for i in range(len(sorted_r_list)):
        if i%200==0:
            reduced_r_list.append(sorted_r_list[i])

    r_smaller=reduced_r_list[0]
    r_interp_list=[]
    for i in range(len(reduced_r_list)): #自适应划分r插值列表
        #print(reduced_r_list[i]-r_smaller)
        if reduced_r_list[i]-r_smaller>R_MIN_INTERVAL:
            #print("YES")
            r_int=(reduced_r_list[i]-r_smaller)/R_DIV_INTERVAL
            for j in range(R_DIV_INTERVAL):
                r_interp_list.append(r_smaller+(j+1)*r_int)

            r_smaller=reduced_r_list[i]
                

    rad_list=np.deg2rad(THETA_LIST);

    ans_list = []
    theta_interp_list = []
    for thetaY in rad_list:
        for thetaZ in rad_list:
            theta_interp_list.append([thetaY,thetaZ])
            for r in r_interp_list:
                x = r/np.sqrt(1+np.tan(thetaY)**2+np.tan(thetaZ)**2)
                y = x*np.tan(thetaY)
                z = x*np.tan(thetaZ)
                ans_list.append([x, y, z])
                
    return ans_list, r_interp_list, theta_interp_list
    #这里返回 ans_list 插值点列表, r_interp_list, theta_interp_list 对应的列表

def gen_polar_coor_grid__remove_periphery(x_list, y_list, z_list):
    # 在gen_polar_coor_grid__cubic的基础上去掉对x轴倾角大于RM_ANGLE的点
    cubic_grid_list, r_interp_list, theta_interp_list=gen_polar_coor_grid__cubic(x_list, y_list, z_list)
    splited_cubic_grid_list=np.array_split(cubic_grid_list, len(cubic_grid_list)//len(r_interp_list))
    
    def lift_angle(xyz):
        x=xyz[0]
        y=xyz[1]
        z=xyz[2]
        return np.arctan(np.sqrt(y**2+z**2)/x)
    
    ans_list=[]
    theta_ans_list=[]
    for i in range(len(splited_cubic_grid_list)):
        if lift_angle(splited_cubic_grid_list[i][0])<=np.deg2rad(RM_ANGLE):
            ans_list.extend(splited_cubic_grid_list[i])
            theta_ans_list.append(theta_interp_list[i])
    
    return ans_list,r_interp_list,theta_ans_list

def gen_phitheta_coor_grid(x_list, y_list, z_list):
    #天顶角圆周角式的极坐标点阵网络
    nnd = len(x_list)
    r_list = [np.sqrt(x_list[i]**2+y_list[i]**2+z_list[i]**2)
              for i in range(nnd)]

    sorted_r_list=sorted(r_list)
    reduced_r_list=[]
    for i in range(len(sorted_r_list)):
        if i%200==0:
            reduced_r_list.append(sorted_r_list[i])

    r_smaller=reduced_r_list[0]
    r_interp_list=[]
    for i in range(len(reduced_r_list)): #自适应划分r插值列表
        #print(reduced_r_list[i]-r_smaller)
        if reduced_r_list[i]-r_smaller>R_MIN_INTERVAL:
            #print("YES")
            r_int=(reduced_r_list[i]-r_smaller)/R_DIV_INTERVAL
            for j in range(R_DIV_INTERVAL):
                r_interp_list.append(r_smaller+(j+1)*r_int)

            r_smaller=reduced_r_list[i]
            
    zenith_rad_list=np.deg2rad(ZENITH_LIST);
    azimuth_rad_list=np.deg2rad(AZIMUTH_LIST);

    ans_list = []
    zen_azi_interp_list = []
    for zenith in zenith_rad_list:
        for azimuth in azimuth_rad_list:
            zen_azi_interp_list.append([zenith,azimuth])
            for r in r_interp_list:
                x = r*np.cos(zenith)
                y = r*np.sin(zenith)*np.cos(azimuth)
                z = r*np.sin(zenith)*np.sin(azimuth)
                ans_list.append([x, y, z])
                
    return ans_list, r_interp_list, zen_azi_interp_list
    #这里返回 ans_list 插值点列表, r_interp_list, theta_interp_list 对应的列表
    

def polar_coordinate_interp(x_list, y_list, z_list, val_list):
    # use a gen_..._grid function and evaluate the interpolated value at gridpoints
    # here val_list can be a list of any kind of values with the same length as x_list

    grid_list, r_interp_list, theta_interp_list = gen_polar_coor_grid__remove_periphery(x_list, y_list, z_list)
    # here function are changeable

    r_div=len(r_interp_list)
    
    var_interp = scipy.interpolate.griddata((x_list, y_list, z_list), val_list, grid_list, method='linear')
    return np.array_split(var_interp, len(var_interp)//r_div), r_interp_list, theta_interp_list

def phitheta_coordinate_interp(x_list, y_list, z_list, val_list):
    # use a gen_phitheta_coor_grid function and evaluate the interpolated value at gridpoints
    # here val_list can be a list of any kind of values with the same length as x_list

    grid_list, r_interp_list, zen_azi_interp_list = gen_phitheta_coor_grid(x_list, y_list, z_list)
    # here function are changeable

    r_div=len(r_interp_list)
    
    var_interp = scipy.interpolate.griddata((x_list, y_list, z_list), val_list, grid_list, method='linear')
    return np.array_split(var_interp, len(var_interp)//r_div), r_interp_list, zen_azi_interp_list

def get_spherical_nodes(case: io.Cases, filename: str, tag: str = "rho"): # 整合好的函数，输入case,filename和目标tag，返回经过插值的结果
    if case.program != Multi_Program.multi_3d:
        raise ValueError("This function is only for Multi-3D")
    x_list, y_list, z_list = case.get_coordinate(filename)
    
    val_list_raw=case.get_data_tag(filename, tag)
    if len(val_list_raw)==len(x_list): # 转换为跟x_list一样定义在节点上
        val_list=val_list_raw
    else:
        cn_list = case.get_data_tag("0", "cn")
        val_list = rho_cell2node(cn_list, val_list_raw)
    
    interp_list, r_list, theta_list= polar_coordinate_interp(x_list, y_list, z_list, val_list)
    return interp_list, r_list, theta_list

def get_spherical_nodes_phitheta(case: io.Cases, filename: str, tag: str = "rho"): # 整合好的函数，输入case,filename和目标tag，返回经过插值的结果，天顶角周角用
    if case.program != Multi_Program.multi_3d:
        raise ValueError("This function is only for Multi-3D")
    x_list, y_list, z_list = case.get_coordinate(filename)
    
    val_list_raw=case.get_data_tag(filename, tag)
    if len(val_list_raw)==len(x_list): # 转换为跟x_list一样定义在节点上
        val_list=val_list_raw
    else:
        cn_list = case.get_data_tag("0", "cn")
        val_list = rho_cell2node(cn_list, val_list_raw)
    
    interp_list, r_list, zen_azi_interp_list= phitheta_coordinate_interp(x_list, y_list, z_list, val_list)
    return interp_list, r_list, zen_azi_interp_list

def get_max_along_r(interp_list,r_list): # 取每条轴上的最大值，粗筛，用 get_spherical_nodes 的结果

    max_arg_list = [] # 存最大值位置的序号
    max_value_list=[] # 存最大值
    max_r_list = [] # 存最大值位置的半径

    for i in range(len(interp_list)):
        # 不能出现无值的列表
        max_arg_list.append(np.nanargmax(interp_list[i])) # 存最大值位置的半径
        max_value_list.append(interp_list[i][max_arg_list[-1]]) # 存最大值
        max_r_list.append(r_list[max_arg_list[-1]]) # 存最大值位置的半径

    return max_value_list,max_r_list,max_arg_list

def shock_wave_interval_along_r(rho_interp_list, r_list): # 返回每条轴上rho>RHO_EFFECT_RATIO*RHO_INITIAL_VALUE区域
    LR_r_list=[]
    for rho_r_list in rho_interp_list:
        L_r=0 # 下边界
        R_r=0 # 上边界
        for i in range(len(r_list)):
            if L_r==0 and rho_r_list[i]>RHO_EFFECT_RATIO*RHO_INITIAL_VALUE:
                L_r=r_list[i]
            if L_r!=0 and R_r==0 and rho_r_list[i]<RHO_EFFECT_RATIO*RHO_INITIAL_VALUE:
                R_r=r_list[i]
        LR_r_list.append([L_r,R_r])

    return LR_r_list

def shock_wave_interval_interp(case: io.Cases, filename: str, tag: str = "rho"): # 返回每条轴上几波区域内的插值结果
    rho_interp_list,r_list,theta_list=get_spherical_nodes(case,filename,"rho")
    LR_r_list=shock_wave_interval_along_r(rho_interp_list,r_list)

    # 得到待插值坐标列表
    coor_list=[]
    for i in range(len(theta_list)):
        thetaY=theta_list[i][0]
        thetaZ=theta_list[i][1]
        r_fine_list=np.linspace(LR_r_list[i][0],LR_r_list[i][1],SHOCK_WAVE_DIV)
        for r_fine in r_fine_list:
            x=r_fine/np.sqrt(1+np.tan(thetaY)**2+np.tan(thetaZ)**2)
            y = x*np.tan(thetaY)
            z = x*np.tan(thetaZ)
            coor_list.append([x,y,z])

    # 插值
    x_list, y_list, z_list=case.get_coordinate(filename)
    val_list_raw=case.get_data_tag(filename, tag)
    cn_list = case.get_data_tag("0", "cn")
    if len(val_list_raw)==len(x_list): # 转换为跟x_list一样定义在节点上
        val_list=val_list_raw
    else:
        val_list = rho_cell2node(cn_list, val_list_raw)
    rho_list=rho_cell2node(cn_list,case.get_data_tag(filename, "rho"))
    var_interp = scipy.interpolate.griddata((x_list, y_list, z_list), val_list, coor_list, method='linear')
    rho_interp = scipy.interpolate.griddata((x_list, y_list, z_list), rho_list, coor_list, method='linear') # 更精细的rho插值
    var_interp_splited=np.array_split(var_interp,len(theta_list))
    rho_interp_splited=np.array_split(rho_interp,len(theta_list))

    return var_interp_splited,rho_interp_splited,LR_r_list,theta_list
    
def get_max_along_r_shock_wave_area(interp_list, LR_r_list): # 取每条轴上的最大值，要用shock_wave_interval_interp的结果
    
    max_arg_list = [] #存最大值位置的序号
    max_value_list=[] #存最大值
    max_r_list=[]
    for i in range(len(interp_list)):
        # 不能出现无值的列表
        max_arg_list.append(np.nanargmax(interp_list[i])) # 存最大值位置的半径
        max_value_list.append(interp_list[i][max_arg_list[-1]]) # 存最大值
        r_list=np.linspace(LR_r_list[i][0],LR_r_list[i][1],SHOCK_WAVE_DIV)
        max_r_list.append(r_list[max_arg_list[-1]]) # 存最大值位置的半径
        
    return max_value_list,max_r_list,max_arg_list

def get_integrated_along_r_shock_wave_area(value_interp_list, rho_interp_list, LR_r_list): # 取每条轴上rho>RHO_EFFECT_RATIO*RHO_INITIAL_VALUE_FINE区域（即冲击波区域）
    int_val_list=[] # 存冲击波区域的 value*r 积分值
    for i in range(len(value_interp_list)):
        r_list=np.linspace(LR_r_list[i][0],LR_r_list[i][1],SHOCK_WAVE_DIV)
        
        int_val=0
        for j in range(SHOCK_WAVE_DIV):
            if rho_interp_list[i][j]>RHO_EFFECT_RATIO*RHO_EFFECT_RATIO_FINE:
                int_val+=value_interp_list[i][j]*(LR_r_list[i][1]-LR_r_list[i][0])/SHOCK_WAVE_DIV

        int_val_list.append(int_val)

    return int_val_list

def get_integrated_along_r_all(value_interp_list, r_list, check_interp_list, check_range_pair): # 取每条轴上check_interp_list∈check_range_pair的区域的value积分值
    int_val_list=[] #存积分值
    for i in range(len(value_interp_list)):
        int_val=0
        for j in list(range(len(value_interp_list[i])))[1:]: #这里为了不数组越界，不考虑第一个值（事实上第一个值一般无效）
            if check_range_pair[0]<check_interp_list[i][j] and check_interp_list[i][j]<check_range_pair[1]:
                int_val+=value_interp_list[i][j]*(r_list[j]-r_list[j-1])

        int_val_list.append(int_val)

    return int_val_list

def get_all_filenames(directory): # 获取指定目录下的所有文件名
    """
    参数：
    directory (str): 目录路径

    返回：
    filenames (list): 目录下所有文件名组成的列表
    """
    filenames = []
    # 使用 os.listdir() 获取目录下所有文件和文件夹的名称
    for filename in os.listdir(directory):
        # 使用 os.path.join() 获取文件的完整路径
        full_path = os.path.join(directory, filename)
        # 判断是否为文件
        if os.path.isfile(full_path):
            filenames.append(filename)
    return filenames

def is_numeric_d(filename):#判断一个文件名是否为数字+".d"的形式
    """
    参数：
    filename (str): 文件名

    返回：
    bool: 如果文件名是数字+".d"的形式，则返回 True，否则返回 False
    """
    # 使用正则表达式匹配数字+".d"的形式
    pattern = r'\.d$'
    if re.search(pattern, filename):
        return True
    else:
        return False
        
def filename_time_list(filenames):#将一个filenames字符串列表中的数字相关的字符串提取出，按数字大小排序
    valid_file_list=[]

    for filename in filenames:
        if is_numeric_d(filename):
            valid_file_list.append(filename[:-2])

    file_time_list=[[filename, float(filename)] for filename in valid_file_list]

    sorted_file_time_list = sorted(file_time_list, key=lambda x: x[1])

    return sorted_file_time_list

def function_time_list(case: io.Cases, time_range_pair, func):#在一个case中将在time_range_pair内的所有file的func值取出列成表
    '''
    这里的func形如func(case,filename)
    time_range_pair应为一个二元列表[ini fnl]，代表处理的时间范围
    返回值为一个列表，其中每一个元素为[func所得值,time]
    '''
    filenames=get_all_filenames(case.target_path)
    sorted_file_time_list=filename_time_list(filenames)

    function_time_pair_list=[]
    
    for file_time_pair in sorted_file_time_list:
        if time_range_pair[0]<=file_time_pair[1] and file_time_pair[1]<=time_range_pair[1]:
            function_time_pair_list.append([func(case,file_time_pair[0]),file_time_pair[1]])

    return function_time_pair_list

def smooth_val_time_list(val_time_pair_list,interval=1e-11,kernel_array=[1/50,1/25,2/25,2/25,2/25,2/25,2/25,2/25,2/25,2/25,2/25,2/25,2/25,1/25,1/50]):#将一个数据的时间列表平滑化
    '''
    val_time_pair_list 是一个列表，其中每一个元素为[val,time]
    interval 是时间均匀插值的最小时间间隔
    '''
    time_list=[i[1] for i in val_time_pair_list]
    val_list=[i[0] for i in val_time_pair_list]

    #时间均匀化
    interp_obj=scipy.interpolate.interp1d(time_list,val_list,kind="linear")
    new_time_list=np.linspace(time_list[0],time_list[-1],int(np.round((time_list[-1]-time_list[0])/interval+1)))
    new_val_list=interp_obj(new_time_list)

    #平滑化数据
    smoothed_val_list=np.convolve(new_val_list,kernel_array,mode="same")
    
    return smoothed_val_list,new_time_list

#----------自定义的Judge Function

def judge_func_depo(case: io.Cases):#为depo特制的Judge_func

    def single_file_judge(case: io.Cases,filename: str):
        depo_interp_list, r_list, theta_list=get_spherical_nodes(case,filename,tag='depo')
        int_depo_list=get_integrated_along_r_all(depo_interp_list,r_list,depo_interp_list,[1e18,1e30])
        return np.sqrt(np.var(int_depo_list))/np.mean(int_depo_list)

    val_time_pair_list=function_time_list(case, [0.89e-9,1.11e-9],single_file_judge)
    smoothed_val_list,time_list=smooth_val_time_list(val_time_pair_list)

    index_1em9=0
    for i in range(len(time_list)):
        if np.abs(time_list[i]-1e-9)<1e-9*1e-2:
            index_1em9=i
            break
    if index_1em9==0:
        print("error")
    #print(time_list)
    return smoothed_val_list[index_1em9]

def judge_func_T(case: io.Cases):#为T特制的judge_func

    def single_file_judge(case: io.Cases,filename: str):
        T_interp_list, r_list, theta_list=get_spherical_nodes(case,filename,tag='T')
        max_value_list,max_r_list,max_arg_list=get_max_along_r(T_interp_list,r_list)
        return np.sqrt(np.var(max_r_list))/np.mean(max_r_list)/2+np.sqrt(np.var(max_value_list))/np.mean(max_value_list)/2

    val_time_pair_list=function_time_list(case, [0.89e-9,1.11e-9],single_file_judge)
    smoothed_val_list,time_list=smooth_val_time_list(val_time_pair_list)

    index_1em9=0
    for i in range(len(time_list)):
        if np.abs(time_list[i]-1e-9)<1e-9*1e-2:
            index_1em9=i
            break
    if index_1em9==0:
        print("error")
    return smoothed_val_list[index_1em9]
