import glob
import random
import re
import numpy as np
import shapefile
from Py6S import *
from matplotlib import pyplot as plt
from osgeo import gdal_array, gdal
from tqdm import tqdm


# 这个文件的作用是进行栅格读写 与其他功能
# 遥感影像下载接口 ： 地理空间数据云 http://www.gscloud.cn/
# 国外接口： EarthExplorer https://earthexplorer.usgs.gov/


# reader 类
class landsat_reader(object):
    # 提供landsat8数据所在目录
    def __init__(self, path):
        # 研究区海拔高度
        self.Altitude = 25
        self.files_arr = []
        # 只要7个波段文件 因为在ENVI里也只展示了可见光的七个波段
        self.bans = 7
        first_file_name = glob.glob("{}/*.tif".format(path))[0]
        for i in range(1, self.bans + 1):
            self.files_arr.append(first_file_name.replace('B1', "B" + str(i)))

    # 返回文件索引
    def index(self, i):
        return self.files_arr[i]

    # format_name can choose "GTiff"or "ENVI"
    def mul_com_fun(self, indexs, out_name, format_name="GTiff", mask=1):
        if len(indexs) > 2:
            print("波段合成中")
        data = read_img(self.band(1))
        image = np.zeros((data[3], data[4], len(indexs)), dtype='f4')
        for i in tqdm(range(len(indexs))):
            data2 = read_img(self.band(indexs[i]))
            image[:, :, i] = data2[0]
        # 替换所有为零处的数据
        image = np.choose(image == 0, (image, np.nan))
        # 标记代表要不要进行辐射定标
        if mask == 1:
            image = self.rad_cai(image, indexs)
            image = self.Atmospheric_correction(image, indexs)

        write_img(out_name, image, data[1], data[2])
        del image
        del data

    # 用波段号作为索引
    def band(self, i):
        return self.files_arr[i - 1]

    # 返回整个索引
    def indexs(self):
        return self.files_arr

    # 打印文件名
    def print_filename(self):
        for f in self.files_arr:
            print(f)

    # 读取数组
    def read_img(self, fileindex):
        read_img(self.files_arr[fileindex])

    # 辐射定标
    def rad_cai(self, image, indexs):
        print("辐射定标")
        # 读取'MTL.txt'内容
        with open(self.band(1).replace('B1.TIF', 'MTL.txt')) as f:
            content = f.read()
        # 利用正则表达式匹配参数
        gain = re.findall(r'RADIANCE_MULT_BAND_\d\s=\s(\S*)', content)
        bias = re.findall(r'RADIANCE_ADD_BAND_\d\s=\s(\S*)', content)
        new_image = np.zeros_like(image, dtype=np.float32)
        for i in tqdm(range(len(indexs))):
            new_image[:, :, i] = (float(gain[indexs[i] - 1]) * image[:, :, i] + float(bias[indexs[i] - 1]))
        return new_image

    # 6s模型参数获取
    def py6s(self, index):

        # 一些要手动输入的参数
        # avg海拔高度 单位为千米
        self.Altitude = 0.030
        # 气溶胶模型
        # NoAerosols = 0Continental = 1Maritime = 2Urban = 3 Desert = 5BiomassBurning = 6Stratospheric = 7
        Aerosol_Model = 3
        # 设置 50nm气溶胶光学厚度 从这个网站查找  https://aeronet.gsfc.nasa.gov/cgi-bin/type_piece_of_map_opera_v2_new
        aot550 = 0.271
        # 添加 py6s 预定义的
        wavelength = Add_wavelength()
        # 打开landsat8元数据文档
        with open(self.band(1).replace('B1.TIF', 'MTL.txt')) as f:
            content = f.read()
        # 初始化模型，寻找可执行的exe文件
        s = SixS()
        s.geometry = Geometry.User()
        # 设置太阳天顶角和方位角
        solar_z = re.findall(r'SUN_ELEVATION = (\S*)', content)
        solar_a = re.findall(r'SUN_AZIMUTH = (\S*)', content)
        s.geometry.solar_z = 90 - float(solar_z[0])
        s.geometry.solar_a = float(solar_a[0])
        # 卫星天顶角和方位角
        s.geometry.view_z = 0
        s.geometry.view_a = 0
        # 获取 影像 范围
        b_lat = re.findall(r'CORNER_\w*LAT\w*\s=\s(\S*)', content)
        b_lon = re.findall(r'CORNER_\w*LON\w*\s=\s(\S*)', content)
        # 求取影像中心经纬度
        center_lon = np.mean([float(i) for i in b_lon])
        center_lat = np.mean([float(i) for i in b_lat])
        # print(center_lon)
        # print(center_lat)
        # 正则匹配时间,返回月日
        time = re.findall(r'DATE_ACQUIRED = (\d{4})-(\d\d)-(\d\d)', content)
        # print("成像时间是{}年{}月{}日".format(time[0][0], time[0][1], time[0][2]))
        s.geometry.month = int(time[0][1])
        s.geometry.day = int(time[0][2])
        # 大气模式类型
        if -15 < center_lat <= 15:
            s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.Tropical)
        if 15 < center_lat <= 45:
            if 4 < s.geometry.month <= 9:
                s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.MidlatitudeSummer)
            else:
                s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.MidlatitudeWinter)
        if 45 < center_lat <= 60:
            if 4 < s.geometry.month <= 9:
                s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.SubarcticSummer)
            else:
                s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.SubarcticWinter)

        s.aero_profile = AtmosProfile.PredefinedType(Aerosol_Model)

        # 这个参数不是很明白
        s.ground_reflectance = GroundReflectance.HomogeneousLambertian(0.36)
        # ENVI中这个默认就是40，单位千米 ，听说Py6s中不能用这个
        # s.visibility = 40.0
        # 550nm气溶胶光学厚度
        s.aot550 = aot550
        # 研究区海拔、卫星传感器轨道高度
        s.altitudes = Altitudes()
        s.altitudes.set_target_custom_altitude(self.Altitude)
        # 将传感器高度设置为卫星高度 ，非常疑惑，它怎么知道我卫星高度多少
        s.altitudes.set_sensor_satellite_level()
        """
        PredefinedWavelengths.LANDSAT_OLI_B1
        预定义的波长，根据点后面的关键字查找 ，py6s库里面列出了B1 到 B9的波长
        """
        # 设置b波普响应函数
        s.wavelength = Wavelength(wavelength[index])

        # 下垫面非均一、朗伯体
        s.atmos_corr = AtmosCorr.AtmosCorrLambertianFromReflectance(-0.1)

        # 运行6s大气模型
        s.run()

        xa = s.outputs.coef_xa
        xb = s.outputs.coef_xb
        xc = s.outputs.coef_xc
        x = s.outputs.values
        return xa, xb, xc

    # 大气校正
    # 进入的是一个np数组
    def Atmospheric_correction(self, image, indexs):
        print("大气校正开始")
        new_image = np.zeros_like(image, dtype=np.float32)
        for i in tqdm(range(len(indexs))):
            a, b, c = self.py6s(indexs[i])
            x = image[:, :, i]
            y = a * x - b
            new_image[:, :, i] = y / (1 + y * c) * 10000
        return new_image


def Add_wavelength():
    wavelength = [0, PredefinedWavelengths.LANDSAT_OLI_B1,
                  PredefinedWavelengths.LANDSAT_OLI_B2,
                  PredefinedWavelengths.LANDSAT_OLI_B3,
                  PredefinedWavelengths.LANDSAT_OLI_B4,
                  PredefinedWavelengths.LANDSAT_OLI_B5,
                  PredefinedWavelengths.LANDSAT_OLI_B6,
                  PredefinedWavelengths.LANDSAT_OLI_B7,
                  PredefinedWavelengths.LANDSAT_OLI_B8,
                  PredefinedWavelengths.LANDSAT_OLI_B9,
                  ]
    return wavelength


# 此函数将地图坐标转换为影像坐标，通过gdal的六参数模型
def geotoimagexy(dataset, x, y):
    trans = dataset.GetGeoTransform()
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])
    return np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解


# 读取原图仿射变换参数值
def transform(ori_transform, offset_x, offset_y):
    # 以列表的形式返回变换数组
    ori_transform = list(ori_transform)
    top_left_x = ori_transform[0] + offset_x * ori_transform[1]
    top_left_y = ori_transform[3] + offset_y * ori_transform[5]
    # 根据仿射变换参数计算新图的原点坐标
    ori_transform[0] = top_left_x
    ori_transform[3] = top_left_y
    return ori_transform


# 输入tif文件名
# return im_data, im_proj, im_geotrans, im_height, im_width, im_bands
def read_img(dataset):
    dataset = gdal.Open(dataset)  # 打开文件
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_bands = dataset.RasterCount  # 波段数
    '''
     GDAL仿射矩阵，包含六个参数 im_geotrans[0]为图像左上角的x坐标 ，im_geotrans[3]为左上角的y坐标
     im_geotrans[1] x方向像素分辨率 ， im_geotrans[5] 为y方向的分辨率
     im_geotrans[2]和 im_geotrans[4] 都为旋转角度，图像朝上的话一般值为0
    '''
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()  # 地图投影信息，字符串表示
    im_data = dataset.ReadAsArray()  # 读取栅格图像的像元数组
    # 下面这个读取的是图像第一波段的的矩阵，是个二维矩阵
    im_data_2 = dataset.GetRasterBand(1).ReadAsArray()
    im_dy = dataset.GetRasterBand(1).DataType
    x_ize = dataset.GetRasterBand(1).XSize
    y_ize = dataset.GetRasterBand(1).YSize
    del dataset  # 关闭对象dataset，释放内存
    return im_data, im_proj, im_geotrans, im_height, im_width, im_bands, im_dy, x_ize, y_ize, im_data_2


# 要输入的是数组化的图像
# 生成指定路径的tif图像
def write_img(output, clip, img_prj, img_trans, format_name="GTiff"):
    # 从该区域读出波段数目，区域大小
    # 根据不同数组的维度进行读取
    Is_GDAL_array = clip.shape[0] < 10
    if len(clip.shape) <= 2:
        im_bands, (im_height, im_width) = 1, clip.shape
    else:
        if Is_GDAL_array:
            im_bands, im_height, im_width = clip.shape
        else:
            im_height, im_width, im_bands = clip.shape
    # 获取Tif的驱动，为创建切出来的图文件做准备
    gtif_driver = gdal.GetDriverByName(format_name)
    if 'int8' in clip.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in clip.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    # （第四个参数代表波段数目，最后一个参数为数据类型，跟原文件一致）
    output_tif = gtif_driver.Create(output, im_width, im_height, im_bands, datatype, options=["INTERLEAVE=BAND"])
    output_tif.SetGeoTransform(img_trans)
    output_tif.SetProjection(img_prj)
    print("开始写入")
    if im_bands == 1:
        if len(clip.shape) == 3:
            clip = clip[:, :, 0]
        output_tif.GetRasterBand(1).WriteArray(clip)
    else:
        for i in tqdm(range(im_bands)):
            if Is_GDAL_array:
                output_tif.GetRasterBand(i + 1).WriteArray(clip[i])
            else:
                output_tif.GetRasterBand(i + 1).WriteArray(clip[:, :, i])
    # 写入磁盘
    output_tif.FlushCache()
    # 计算统计信息
    # for i in range(1, im_bands):
    #     output_tif.GetRasterBand(i).ComputeStatistics(False)
    # # 创建概视图或金字塔图层
    # output_tif.BuildOverviews('average', [2, 4, 8, 16, 32])
    # 关闭output文件
    del output_tif
    print("成功写入" + output)


# 三个参数依次为 输入的tif路径 ， 输出路径 ， 用于裁剪的shp文件路径
def clip_function(input, output, shp):
    # 打开需要裁剪的遥感影像
    input_tif = gdal.Open(input)
    # 读取裁剪的shape文件的外接矩形大小
    shp = shapefile.Reader(shp)
    minX, minY, maxX, maxY = shp.bbox
    # 定义切图的起始点和终点坐标(相比原点的横坐标和纵坐标偏移量)
    offset_x, offset_y = geotoimagexy(input_tif, minX, minY)
    endset_x, endset_y = geotoimagexy(input_tif, maxX, maxY)
    # 定义切图的大小（矩形框）
    block_xsize = int(endset_x - offset_x + 1)
    block_ysize = int(abs(endset_y - offset_y) + 1)
    # 将裁剪区域的影像转换为数组
    # 需要注意的是图像原点在左上角，因此我们要将原点的y向上移动，因为y轴向下，所以符号为减
    offset_y = offset_y - block_ysize
    clip = input_tif.ReadAsArray(offset_x, offset_y, block_xsize, block_ysize)
    img_prj = input_tif.GetProjection()
    img_trans = transform(input_tif.GetGeoTransform(), offset_x, offset_y)
    # 调用保存函数
    print("开始裁剪")
    write_img(output, clip, img_prj, img_trans, "GTiff")


# 获取ndvi ，返回数组 ，去除异常值
def getndvi(nir_data, red_data):
    try:
        # 将红外数组与近红外数组相加，得到新数组，每一个行列确定的像元上，都会带有一个像素信息，这些就是波段存储的值
        '''
        np里面的数组运算需要行列相等，也就是shape相等，但是对于行或者列向量可以豁免，计算时将扩充为同型矩阵 ，矩阵的数学乘法用np.dot()函数
        矩阵乘法，前行X后列，需要满足前列等于后行
        '''
        denominator = np.array(nir_data + red_data, dtype='f4')
        numerator = np.array(nir_data - red_data, dtype='f4')
        ndvi = np.divide(numerator, denominator, where=denominator != 0.0)
        # 去除异常值 使得ndvi的值在0与1之间
        ndvi = np.choose((ndvi < 0) + 2 * (ndvi > 1), (ndvi, 0, 0))
        return ndvi

    except BaseException as e:
        print(str(e))


# 此函数进入的是gdal类型数组
# 包括多波段与单波段
def linear_stretch(gray, truncated_value, max_out=255, min_out=0):
    truncated_down = np.percentile(gray, truncated_value)
    truncated_up = np.percentile(gray, 100 - truncated_value)
    gray = (gray - truncated_down) / (truncated_up - truncated_down) * (max_out - min_out) + min_out
    gray = np.choose(gray < min_out + 2 * (gray > max_out), (gray, min_out, max_out))
    gray = np.uint8(gray)
    return gray


# 传入tif文件
def show_hist(src):
    # 直方图
    src_array = read_img(src)[0]
    # 灰度拉伸
    if src_array.shape[0] < 10:
        src_array = linear_stretch(src_array[0, :, :], 2)
    print(src_array)
    plt.hist(src_array)
    plt.title("Single_band_histogram of %s" % src)
    plt.xlabel("x axis caption")
    plt.ylabel("y axis caption")
    print("Histogram display")
    plt.show()


# Image classification by single band threshold method
# Input parameter TIF file, target classification picture, threshold
def classification(src, tgt, threshold):
    print("Image classification start")
    threshold = list(threshold)
    src_data = read_img(src)
    src_array = np.array(src_data[0])
    # 灰度拉伸
    if src_array.shape[0] < 10:
        src_array = linear_stretch(src_array[0, :, :], 2)
    Max = np.max(src_array)
    Min = np.min(src_array)
    print("The maximum value of data is%s" % Max)
    print("The minimum value of data is%s" % Min)
    print(src_array)
    # 保存提取结果
    # rgb = Create_color_slice(src_array)
    color = [0, 0, 175]
    # 掩膜
    mask = gdal_array.numpy.logical_and(src_array > threshold[0], src_array < threshold[1])
    rgb = gdal_array.numpy.zeros((3, src_array.shape[0], src_array.shape[1],), gdal_array.numpy.uint8)
    for i in range(3):
        rgb[i] = np.choose(mask, (255, color[i]))
    output = gdal_array.SaveArray(rgb.astype(gdal_array.numpy.uint8), tgt + ".jpg", format="JPEG")
    del output
    del src_data


# 输入np数组
def np_conversion_gdal(array):
    if len(array.shape) == 2:
        gdal_arr = array
    else:
        gdal_arr = gdal_array.numpy.zeros((array.shape[2], array.shape[0], array.shape[1],), gdal_array.numpy.uint8)
    for i in range(array.shape[2]):
        gdal_arr[i] = array[:, :, i]
    return gdal_arr


# 输入gdal类型数组
def gdal_conversion_np(array):
    array2 = array.shape
    np_arr = np.zeros(shape=(array2[1], array2[2], array2[0]))
    for i in range(array2[0]):
        np_arr[:, :, i] = array[i]
    return np_arr


# 输入数组 返回一个gdal类型的数组 ,输入的是灰度直方图
def Create_color_slice(array, bins=10):
    # 获取区间
    # random.randint(0, 255)
    classes = gdal_array.numpy.histogram(array, bins=bins)[1]
    rgb = gdal_array.numpy.zeros((3, array.shape[0], array.shape[1],), gdal_array.numpy.uint8)
    for i in range(bins):
        mask = gdal_array.numpy.logical_and(array > classes[i], array < classes[i + 1])
        for j in tqdm(range(3)):
            rgb[j] = np.choose(mask, (rgb[j], random.randint(0, 255)))
    rgb = rgb.astype(gdal_array.numpy.uint8)
    return rgb


# 引入写好的文件
from landsat8_fun import *
from Py6S import SixS

# 主函数
if __name__ == "__main__":
    # 这行代码测试你的6s有没有正常运行
    # 如果你的py6s 库找不到路径请注释掉 辐射定标和大气校正的代码
    SixS.test()
    print("主函数开始运行")
    # bind_math("(a-b)/(c+d)", 1, 5, 25, 29)
    # 读取路径下的landsat文件夹
    file = landsat_reader("./LC81210382021028LGN00")
    # 先对第4波段和第5波段进行辐射校正和大气校正，第一个参数是一个列表，数字代表landsat第几个波段，一旦长度超过二，就会进行波段融合
    # 如果你的py6s安装不上注释下述代码
    file.mul_com_fun([3], "./测试结果/green.tif")
    file.mul_com_fun([6], "./测试结果/nir.tif")
    # 根据裁剪文件进行裁剪
    clip_function("./测试结果/green.tif", "./测试结果/landsat_green.tif", "裁剪用数据/utm50n_mask.shp")
    clip_function("./测试结果/nir.tif", "./测试结果/landsat_nir.tif", "裁剪用数据/utm50n_mask.shp")
    # 跳过辐射定标和大气校正
    # clip_function(file.band(4), "./测试结果/landsat_green.tif", "裁剪用数据/utm50n_mask.shp")
    # clip_function(file.band(5), "./测试结果/landsat_nir.tif", "裁剪用数据/utm50n_mask.shp")
    # 读取裁剪后文件
    green_signal = read_img("./测试结果/landsat_green.tif")
    nir_signal = read_img("./测试结果/landsat_nir.tif")
    # 计算植被指数
    ndsi = getndvi(green_signal[0], nir_signal[0])
    write_img("./测试结果/ndsi.tif", ndsi, green_signal[1], green_signal[2])