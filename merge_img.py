from osgeo import osr, gdal
import os 
 
 
def assign_spatial_reference_byfile(src_path, dst_path):
    src_ds = gdal.Open(src_path, gdal.GA_ReadOnly)
    print(src_ds.GetProjectionRef())
    sr = osr.SpatialReference()
    sr.ImportFromWkt(src_ds.GetProjectionRef())
    geoTransform = src_ds.GetGeoTransform()
    dst_ds = gdal.Open(dst_path, gdal.GA_Update)
    dst_ds.SetProjection(sr.ExportToWkt())
    dst_ds.SetGeoTransform(geoTransform)
    dst_ds = None
    src_ds = None




for i in os.listdir('patch/gt'):
    
    src_path = os.path.join('patch',os.path.join('gt', i)) 
    dst_path = os.path.join('pred',i) 
    print(dst_path)
    assign_spatial_reference_byfile(src_path, dst_path)

