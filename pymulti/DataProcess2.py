from .CaseIO import Cases
import scipy.interpolate
import numpy as np
from .utils import Multi_Program
import matplotlib.pyplot as plt

# 全局变量
RM_ANGLE = 40.0  # 边缘仰角，大于这个角度的样本将不被考虑，设置为金锥内壁的角度
THETA_DIV = 41  # 角方向的切分段数
R_MIN_INTERVAL = 0.001  # 径向最小被划分区间长度
R_DIV_INTERVAL = 5  # 径向每个被划分区间被划分出的段数
RHO_INITIAL_VALUE = 1  # 未被压缩区域的密度
RHO_EFFECT_RATIO = 1.1  # 密度大于 RHO_EFFECT_RATIO*RHO_INITIAL_VALUE 的区域被考虑
RHO_EFFECT_RATIO_FINE = 1.2  # 密度大于 RHO_EFFECT_RATIO_FINE*RHO_INITIAL_VALUE 的区域被计算入积分
SHOCK_WAVE_DIV = 100  # 激波区的切分段数

THETA_LIST = [-RM_ANGLE+i*2*RM_ANGLE/(THETA_DIV-1)
              for i in range(THETA_DIV)]  # 角方向差值点
THETA_CENTER = int((THETA_DIV-1)/2)  # 中央射线的序号


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
            file.write("%.6f\n" % num)  # 将浮点数格式化为字符串，并写入文件


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

    sorted_r_list = sorted(r_list)
    reduced_r_list = []
    for i in range(len(sorted_r_list)):
        if i % 200 == 0:
            reduced_r_list.append(sorted_r_list[i])

    r_smaller = reduced_r_list[0]
    r_interp_list = []
    for i in range(len(reduced_r_list)):  # 自适应划分r插值列表
        # print(reduced_r_list[i]-r_smaller)
        if reduced_r_list[i]-r_smaller > R_MIN_INTERVAL:
            # print("YES")
            r_int = (reduced_r_list[i]-r_smaller)/R_DIV_INTERVAL
            for j in range(R_DIV_INTERVAL):
                r_interp_list.append(r_smaller+(j+1)*r_int)

            r_smaller = reduced_r_list[i]

    rad_list = np.deg2rad(THETA_LIST)

    ans_list = []
    theta_interp_list = []
    for thetaY in rad_list:
        for thetaZ in rad_list:
            theta_interp_list.append([thetaY, thetaZ])
            for r in r_interp_list:
                x = r/np.sqrt(1+np.tan(thetaY)**2+np.tan(thetaZ)**2)
                y = x*np.tan(thetaY)
                z = x*np.tan(thetaZ)
                ans_list.append([x, y, z])

    return ans_list, r_interp_list, theta_interp_list
    # 这里返回 ans_list 插值点列表, r_interp_list, theta_interp_list 对应的列表


def gen_polar_coor_grid__remove_periphery(x_list, y_list, z_list):
    # 在gen_polar_coor_grid__cubic的基础上去掉对x轴倾角大于RM_ANGLE的点
    cubic_grid_list, r_interp_list, theta_interp_list = gen_polar_coor_grid__cubic(
        x_list, y_list, z_list)
    splited_cubic_grid_list = np.array_split(
        cubic_grid_list, len(cubic_grid_list)//len(r_interp_list))

    def lift_angle(xyz):
        x = xyz[0]
        y = xyz[1]
        z = xyz[2]
        return np.arctan(np.sqrt(y**2+z**2)/x)

    ans_list = []
    theta_ans_list = []
    for i in range(len(splited_cubic_grid_list)):
        if lift_angle(splited_cubic_grid_list[i][0]) <= np.deg2rad(RM_ANGLE):
            ans_list.extend(splited_cubic_grid_list[i])
            theta_ans_list.append(theta_interp_list[i])

    return ans_list, r_interp_list, theta_ans_list


def polar_coordinate_interp(x_list, y_list, z_list, val_list):
    # use a gen_..._grid function and evaluate the interpolated value at gridpoints
    # here val_list can be a list of any kind of values with the same length as x_list

    grid_list, r_interp_list, theta_interp_list = gen_polar_coor_grid__remove_periphery(
        x_list, y_list, z_list)
    # here function are changeable

    r_div = len(r_interp_list)

    var_interp = scipy.interpolate.griddata(
        (x_list, y_list, z_list), val_list, grid_list, method='linear')
    return np.array_split(var_interp, len(var_interp)//r_div), r_interp_list, theta_interp_list


# 整合好的函数，输入case,filename和目标tag，返回经过插值的结果
def get_spherical_nodes(case: Cases, filename: str, tag: str = "rho"):
    if case.program != Multi_Program.multi_3d:
        raise ValueError("This function is only for Multi-3D")
    x_list, y_list, z_list = case.get_coordinate(filename)

    val_list_raw = case.get_data_tag(filename, tag)
    if len(val_list_raw) == len(x_list):  # 转换为跟x_list一样定义在节点上
        val_list = val_list_raw
    else:
        cn_list = case.get_data_tag("0", "cn")
        val_list = rho_cell2node(cn_list, val_list_raw)

    interp_list, r_list, theta_list = polar_coordinate_interp(
        x_list, y_list, z_list, val_list)
    return interp_list, r_list, theta_list


# 返回每条轴上rho>RHO_EFFECT_RATIO*RHO_INITIAL_VALUE区域
def shock_wave_interval_along_r(rho_interp_list, r_list):
    LR_r_list = []
    for rho_r_list in rho_interp_list:
        L_r = 0  # 下边界
        R_r = 0  # 上边界
        for i in range(len(r_list)):
            if L_r == 0 and rho_r_list[i] > RHO_EFFECT_RATIO*RHO_INITIAL_VALUE:
                L_r = r_list[i]
            if L_r != 0 and R_r == 0 and rho_r_list[i] < RHO_EFFECT_RATIO*RHO_INITIAL_VALUE:
                R_r = r_list[i]
        LR_r_list.append([L_r, R_r])

    return LR_r_list


# 返回每条轴上几波区域内的插值结果
def shock_wave_interval_interp(case: Cases, filename: str, tag: str = "rho"):
    rho_interp_list, r_list, theta_list = get_spherical_nodes(
        case, filename, "rho")
    LR_r_list = shock_wave_interval_along_r(rho_interp_list, r_list)

    # 得到待插值坐标列表
    coor_list = []
    for i in range(len(theta_list)):
        thetaY = theta_list[i][0]
        thetaZ = theta_list[i][1]
        r_fine_list = np.linspace(
            LR_r_list[i][0], LR_r_list[i][1], SHOCK_WAVE_DIV)
        for r_fine in r_fine_list:
            x = r_fine/np.sqrt(1+np.tan(thetaY)**2+np.tan(thetaZ)**2)
            y = x*np.tan(thetaY)
            z = x*np.tan(thetaZ)
            coor_list.append([x, y, z])

    # 插值
    x_list, y_list, z_list = case.get_coordinate(filename)
    val_list_raw = case.get_data_tag(filename, tag)
    cn_list = case.get_data_tag("0", "cn")
    if len(val_list_raw) == len(x_list):  # 转换为跟x_list一样定义在节点上
        val_list = val_list_raw
    else:
        val_list = rho_cell2node(cn_list, val_list_raw)
    rho_list = rho_cell2node(cn_list, case.get_data_tag(filename, "rho"))
    var_interp = scipy.interpolate.griddata(
        (x_list, y_list, z_list), val_list, coor_list, method='linear')
    rho_interp = scipy.interpolate.griddata(
        (x_list, y_list, z_list), rho_list, coor_list, method='linear')  # 更精细的rho插值
    var_interp_splited = np.array_split(var_interp, len(theta_list))
    rho_interp_splited = np.array_split(rho_interp, len(theta_list))

    return var_interp_splited, rho_interp_splited, LR_r_list, theta_list


# 取每条轴上的最大值，要用shock_wave_interval_interp的结果
def get_max_along_r(interp_list, LR_r_list):

    max_arg_list = []  # 存最大值位置的序号
    max_value_list = []  # 存最大值
    max_r_list = []
    for i in range(len(interp_list)):
        # 不能出现无值的列表
        max_arg_list.append(np.nanargmax(interp_list[i]))  # 存最大值位置的半径
        max_value_list.append(interp_list[i][max_arg_list[-1]])  # 存最大值
        r_list = np.linspace(LR_r_list[i][0], LR_r_list[i][1], SHOCK_WAVE_DIV)
        max_r_list.append(r_list[max_arg_list[-1]])  # 存最大值位置的半径

    return max_value_list, max_r_list, max_arg_list


# 取每条轴上rho>RHO_EFFECT_RATIO*RHO_INITIAL_VALUE_FINE区域（即冲击波区域）
def get_integrated_along_r(value_interp_list, rho_interp_list, LR_r_list):
    int_val_list = []  # 存冲击波区域的 value*r 积分值
    for i in range(len(value_interp_list)):
        r_list = np.linspace(LR_r_list[i][0], LR_r_list[i][1], SHOCK_WAVE_DIV)

        int_val = 0
        for j in range(SHOCK_WAVE_DIV):
            if rho_interp_list[i][j] > RHO_EFFECT_RATIO*RHO_EFFECT_RATIO_FINE:
                int_val += value_interp_list[i][j] * \
                    (LR_r_list[i][1]-LR_r_list[i][0])/SHOCK_WAVE_DIV

        int_val_list.append(int_val)

    return int_val_list
