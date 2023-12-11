import CaseIO as io
import scipy.interpolate
import numpy as np
from . import Multi_Program


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


def gen_polar_coor_grid__cubic(x_list, y_list, z_list, r_div=125, thetaZ_div=25, thetaY_div=25):
    # see readme
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


def polar_coordinate_interp(x_list, y_list, z_list, val_list, r_div=125, thetaZ_div=25, thetaY_div=25):
    # use a gen_..._grid function and evaluate the interpolated value at gridpoints
    # here val_list can be a list of any kind of values with the same length as x_list

    grid_list, r_interp_list = gen_polar_coor_grid__cubic(
        x_list, y_list, z_list, r_div, thetaZ_div, thetaY_div)
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

