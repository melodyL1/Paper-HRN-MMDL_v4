from TPPI.models.TPPP import DL_Shallow_Network, CNN_1D, CNN_2D, CNN_3D, HybridSN, SSAN, pResNet, VGG16, MultiModelCNN, HRN, NEW
from TPPI.models.SSRNSSTNNet import SSNet_AEAE_IN, SSRN
from TPPI.models.SSFTTnet import SSFTTnet,SSRNTransNEWMobil,SSRNTransNEWMobil06LiDAR,TransHSI
from TPPI.models.nextvit import NextViT,SSRNNextViT
from TPPI.models.cmt import CMT,CMT3DCNN
from TPPI.models.Google_ViT import VisionTransformer
from TPPI.models.vit_pytorch.mobile_vit import MobileViT, MobileViT_Improve
from TPPI.models.MTUNet import MTUHSINet
from TPPI.models.NewModels import MultiModelTrans

def get_model(modelName, dataset):
    model = get_model_instance(modelName)
    model = model(dataset)
    return model

def get_model_instance(name):
    try:
        return {
            'Shallow_Network': DL_Shallow_Network,
            'CNN_1D': CNN_1D,
            'CNN_2D': CNN_2D,
            'VGG16': VGG16,
            'CNN_3D': CNN_3D,
            'HybridSN': HybridSN,
            'pResNet': pResNet,
            'SSRN': SSRN,
            'SSTN': SSNet_AEAE_IN,
            'SSAN': SSAN,
            'SSRNNextViT':SSRNNextViT,
            'SSFTTnet': SSFTTnet,
            'SSRNTransNEWMobil': SSRNTransNEWMobil,
            'SSRNTransNEWMobil06LiDAR': SSRNTransNEWMobil06LiDAR,
            'CMT': CMT,
            'CMT3DCNN': CMT3DCNN,
            'VisionTransformer': VisionTransformer,
            'NextViT': NextViT,
            'MobileViT':MobileViT,
            'MobileViT_Improve':MobileViT_Improve,
            'MTUHSINet': MTUHSINet,
            'TransHSI':TransHSI,
            'MultiModelCNN':MultiModelCNN,
            'HRN':HRN,
            'NEW':MultiModelTrans
        }[name]
    except:
        raise ("Model {} not available".format(name))


