import xml.etree.ElementTree as ET
import geopandas as gpd
from shapely.geometry import Polygon
import os
from osgeo import gdal, osr

def get_image_crs_and_affine(tif_path):
    """
    从TIFF影像中获取坐标参考系（CRS）和仿射变换参数。

    参数:
        tif_path: 遥感影像文件路径 (.tif)

    返回:
        crs: geopandas 可用的 CRS 对象
        affine: 仿射变换参数元组 (top_left_x, x_resolution, x_rotation, top_left_y, y_rotation, y_resolution)
    """
    dataset = gdal.Open(tif_path, gdal.GA_ReadOnly)
    if not dataset:
        raise FileNotFoundError(f"无法打开影像文件: {tif_path}")

    # 获取仿射变换参数
    affine = dataset.GetGeoTransform()
    if not affine or (affine[0] == 0 and affine[3] == 0):
         raise ValueError(f"影像文件 {tif_path} 不包含有效的地理变换信息。")

    # 获取坐标参考系 (CRS)
    proj = osr.SpatialReference(wkt=dataset.GetProjection())
    crs = proj.ExportToProj4()

    dataset = None  # 关闭文件
    return crs, affine

def pixel_to_geo_coords(pixel_x, pixel_y, affine):
    """
    使用仿射变换参数将像素坐标转换为地理坐标。

    参数:
        pixel_x: 像素列号 (x)
        pixel_y: 像素行号 (y)
        affine: 从 get_image_crs_and_affine 函数获取的仿射参数元组

    返回:
        (geo_x, geo_y): 转换后的地理坐标
    """
    # 仿射变换公式:
    # Xgeo = GT(0) + Xpixel*GT(1) + Yline*GT(2)
    # Ygeo = GT(3) + Xpixel*GT(4) + Yline*GT(5)
    geo_x = affine[0] + pixel_x * affine[1] + pixel_y * affine[2]
    geo_y = affine[3] + pixel_x * affine[4] + pixel_y * affine[5]
    return (geo_x, geo_y)

def xml_to_shp_with_image_ref(xml_path, output_dir, tif_path, is_xml_pixel_coords=True):
    """
    将XML格式的区域数据转换为SHP格式，并利用参考影像确保坐标对齐。

    参数:
        xml_path: XML文件路径
        output_dir: 输出SHP文件的目录
        tif_path: 参考遥感影像路径 (.tif)
        is_xml_pixel_coords: 如果XML中的坐标是像素坐标，则为True；如果已经是地理坐标，则为False。
    """
    # 1. 从参考影像获取坐标信息
    print(f"正在从影像 {tif_path} 读取坐标信息...")
    crs, affine = get_image_crs_and_affine(tif_path)
    print("坐标信息读取成功。")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 解析XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    features = []

    # 遍历每个Region
    for region in root.findall('Region'):
        region_name = region.get('name', 'Unknown')
        region_color = region.get('color', 'Unknown')

        geometry_def = region.find('GeometryDef')
        if geometry_def is None:
            continue

        for polygon in geometry_def.findall('Polygon'):
            coord_elem = polygon.find('Exterior/LinearRing/Coordinates')
            if coord_elem is None:
                continue
            coord_str = coord_elem.text.strip()

            coord_parts = [float(p) for p in coord_str.split() if p]
            points = []
            for i in range(0, len(coord_parts), 2):
                if i + 1 < len(coord_parts):
                    x = coord_parts[i]
                    y = coord_parts[i + 1]

                    # 2. 根据坐标类型决定是否转换
                    if is_xml_pixel_coords:
                        # 如果是像素坐标，进行转换
                        geo_x, geo_y = pixel_to_geo_coords(x, y, affine)
                        points.append((geo_x, geo_y))
                    else:
                        # 如果已经是地理坐标，直接使用
                        points.append((x, y))

            # 创建多边形（确保闭合）
            if len(points) >= 3:
                if points[0] != points[-1]:
                    points.append(points[0])
                # 过滤掉可能因转换产生的无效坐标
                if not any(p[0] is None or p[1] is None for p in points):
                    polygon_geom = Polygon(points)
                    features.append({
                        'geometry': polygon_geom,
                        'name': region_name,
                        'color': region_color
                    })

    # 3. 创建GeoDataFrame，并应用从影像获取的CRS
    if not features:
        print("警告：未从XML中解析到任何有效的多边形要素。")
        return

    gdf = gpd.GeoDataFrame(features, crs=crs)

    # 导出为SHP文件
    output_path = os.path.join(output_dir, 'regions333.shp')
    gdf.to_file(output_path, driver='ESRI Shapefile', encoding='utf-8')
    print(f"SHP文件已成功导出至: {output_path}")
    print(f"SHP文件坐标系已与参考影像 {tif_path} 对齐。")

# --- 使用示例 ---
if __name__ == "__main__":
    XML_FILE = "yangben20251122.xml"
    OUTPUT_DIR = "shp_output"
    TIF_FILE = r"F:\LXM\GF_ZZ152024\1gf2015-2024caijian\2015subset.tif"  # 你的遥感影像路径

    # !!! 关键：请根据你的XML坐标类型选择下面的参数 !!!
    # 如果你的XML中的坐标是像素行列号（最常见于ENVI未配准影像的标注），请使用 True
    # 如果你的XML中的坐标已经是UTM等地理坐标，请使用 False
    XML_COORDS_ARE_PIXELS = True  # <--- 修改这里

    try:
        xml_to_shp_with_image_ref(XML_FILE, OUTPUT_DIR, TIF_FILE, XML_COORDS_ARE_PIXELS)
    except Exception as e:
        print(f"转换过程中出现错误: {e}")