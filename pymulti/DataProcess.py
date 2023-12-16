import CaseIO as io
import scipy.interpolate
import numpy as np
from __init__ import Multi_Program

R_DIV=125 #径向细分段数，全局变量，有需要请在此处改动
RM_ANGLE=40 #边缘仰角，大于这个角度的样本将不被考虑，**角度值**，全局变量，有需要请在此处改动
            #这里改成40是为了去除不可靠边界的影响，本应为50

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


def gen_polar_coor_grid__cubic(x_list, y_list, z_list, r_div=R_DIV, thetaZ_div=25, thetaY_div=25):
    # 方形的一个极坐标网格点阵，**不要给r_div赋值！！**，在全局变量处改动
    nnd = len(x_list)
    r_list = [np.sqrt(x_list[i]**2+y_list[i]**2+z_list[i]**2)
              for i in range(nnd)]
    thetaZ_list = [np.arctan(z_list[i]/x_list[i]) for i in range(nnd)]
    thetaY_list = [np.arctan(y_list[i]/x_list[i]) for i in range(nnd)]

    rMin = min(r_list)
    rMax = max(r_list)
    thetaZMin = min(thetaZ_list)
    thetaZMax = max(thetaZ_list)
    thetaYMin = min(thetaY_list)
    thetaYMax = max(thetaY_list)

    # print(rMin,rMax,thetaYMin,thetaYMax,thetaZMin,thetaZMax)

    r_interp_list = np.linspace(rMin, rMax, r_div)
    thetaY_interp_list = np.linspace(thetaYMin, thetaYMax, thetaY_div)
    thetaZ_interp_list = np.linspace(thetaZMin, thetaZMax, thetaZ_div)

    ans_list = []
    for thetaY in thetaY_interp_list:
        for thetaZ in thetaZ_interp_list:
            for r in r_interp_list:
                x = r/np.sqrt(1+np.tan(thetaY)**2+np.tan(thetaZ)**2)
                y = x*np.tan(thetaY)
                z = x*np.tan(thetaZ)
                ans_list.append([x, y, z])

    return ans_list, r_interp_list

def gen_polar_coor_grid__remove_periphery(x_list, y_list, z_list, 
                                          rm_angle=RM_ANGLE, r_div=R_DIV, thetaZ_div=25, thetaY_div=25):
    # 在gen_polar_coor_grid__cubic的基础上去掉对x轴倾角大于rm_angle的点
    # rm_angle 输入**角度值**，**不要给rm_angle赋值！！**，在全局变量处改动
    # **不要给r_div赋值！！**，在全局变量处改动
    cubic_grid_list, r_interp_list=gen_polar_coor_grid__cubic(x_list, y_list, z_list, thetaZ_div=25, thetaY_div=25)
    splited_cubic_grid_list=np.array_split(cubic_grid_list, len(cubic_grid_list)//r_div)
    
    def lift_angle(xyz):
        x=xyz[0]
        y=xyz[1]
        z=xyz[2]
        return np.arctan(np.sqrt(y**2+z**2)/x)
    
    ans_list=[]
    for i in range(len(splited_cubic_grid_list)):
        if lift_angle(splited_cubic_grid_list[i][0])<=np.deg2rad(rm_angle):
            ans_list.extend(splited_cubic_grid_list[i])
    
    return ans_list,r_interp_list    

def polar_coordinate_interp(x_list, y_list, z_list, val_list, r_div=R_DIV, thetaZ_div=25, thetaY_div=25):
    # use a gen_..._grid function and evaluate the interpolated value at gridpoints
    # here val_list can be a list of any kind of values with the same length as x_list
    # **不要给r_div赋值！！**，在全局变量处改动

    grid_list, r_interp_list = gen_polar_coor_grid__remove_periphery(
        x_list, y_list, z_list, RM_ANGLE, r_div, thetaZ_div, thetaY_div)
    # here function are changeable
    ng = r_div*thetaZ_div*thetaY_div
    var_interp = scipy.interpolate.griddata(
        (x_list, y_list, z_list), val_list, grid_list, method='linear')
    return np.array_split(var_interp, len(var_interp)//r_div), r_interp_list


def get_spherical_nodes(case: io.Cases, filename: str, tag: str = "rho"):
    if case.program != Multi_Program.multi_3d:
        raise ValueError("This function is only for Multi-3D")
    x_list, y_list, z_list = case.get_coordinate(filename)
    cn_list = case.get_data_tag("0", "cn")
    
    val_list_raw=case.get_data_tag(filename, tag)
    if len(val_list_raw)==len(x_list): # 转换为跟x_list一样定义在节点上
        val_list=val_list_raw
    else:
        val_list = rho_cell2node(cn_list, val_list_raw)
    
    interp_list, r_list = polar_coordinate_interp(
        x_list, y_list, z_list, val_list)
    return interp_list, r_list


def get_max_sphere_along_r_3(case: io.Cases, filename: str, tag: str = "rho"):
    if case.program != Multi_Program.multi_3d:
        raise ValueError("This function is only for Multi-3D")
    interp_list, r_list = get_spherical_nodes(case, filename, tag)
    max_arg_list = [] #存最大值位置的序号
    for i in range(len(interp_list)):
        if not (np.isnan(interp_list[i]).all()):
            max_arg_list.append(np.nanargmax(interp_list[i]))
    max_r_list=[r_list[i] for i in max_arg_list] #存最大值位置的半径
    return max_r_list,max_arg_list

def get_max_value_along_r_3(case: io.Cases, filename: str, tag: str = "rho"):
    # 取每条轴上的最大值
    if case.program != Multi_Program.multi_3d:
        raise ValueError("This function is only for Multi-3D")
    interp_list, r_list = get_spherical_nodes(case, filename, tag)
    max_arg_list = [] #存最大值位置的序号
    max_value_list=[] #存最大值
    for i in range(len(interp_list)):
        if not (np.isnan(interp_list[i]).all()):
            max_arg_list.append(np.nanargmax(interp_list[i]))
            max_value_list.append(interp_list[i][max_arg_list[-1]])
    max_r_list=[r_list[i] for i in max_arg_list] #存最大值位置的半径
    return max_value_list,max_r_list,max_arg_list

def visual_surface_data_list(case: io.Cases, filename: str, tag: str = "rho", surface_func = np.nanargmax):
    # 用于返回可用于可视化的特征点列表。列表中每个元素为特征点的x,y,z
    if case.program != Multi_Program.multi_3d:
        raise ValueError("This function is only for Multi-3D")
    
    r_div=R_DIV
    interp_list, *rest = get_spherical_nodes(case, filename, tag)
    grid_list, *rest = gen_polar_coor_grid__remove_periphery(case.get_data_tag(filename,"x"),
                                                             case.get_data_tag(filename,"y"),
                                                             case.get_data_tag(filename,"z"))
    grid_list_split=np.array_split(grid_list, len(grid_list)//r_div)
    ans_list=[]
    for i in range(len(interp_list)):
        if not (np.isnan(interp_list[i]).all()):
            ans_list.append(grid_list_split[i][surface_func(interp_list[i])])
    max_r_list, *rest=get_max_sphere_along_r_3(case, filename, tag)
    return ans_list,max_r_list

def heterogeneity_quantization(case: io.Cases, filename: str, tag: str = "T", weight_r=0.7, weight_v=0.3):
    #测量两个标准差平均值比：tag最大面位置和tag最大点值，分别以wight_r,weight_v加权后求和返回
    max_value_list,max_r_list,*rest=get_max_value_along_r_3(case,filename,tag)
    j_v=np.sqrt(np.var(max_value_list))/np.mean(max_value_list)
    j_r=np.sqrt(np.var(max_r_list))/np.mean(max_r_list)
    return weight_r*j_r+weight_v*j_v,j_v,j_r
    
    
