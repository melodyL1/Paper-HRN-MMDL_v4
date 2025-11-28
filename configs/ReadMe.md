* GPU: value, GPU to be used
* Model: name. [Options: 'Shallow_Network',
                         'CNN_1D','CNN_2D','CNN_3D','VGG16','HybridSN','pResNet',
                         'SSRN','SSTN','SSAN','SSRNNextViT','SSFTTnet',
                         'SSRNTransNEWMobil','SSRNTransNEWMobil06LiDAR',
                         'CMT','CMT3DCNN',
                         'VisionTransformer','NextViT' ,'MobileViT','MobileViT_Improve','MTUHSINet']
* New Model: name. [Options: 'TransHSI'
                    Options_Multimodal: 'MultiModelCNN','HRN','NEW']
* Classic algorithms optional: name. ['Shallow_Network',
                                      'CNN_1D','CNN_2D','CNN_3D','VGG16','HybridSN',
                                      'SSTN','SSFTTnet',
                                      'CMT',
                                      'VisionTransformer','NextViT',
                                      'TransHSI','HRN']
* Test period model：: name. ['MultiModelTrans']
* 注意事项：（1）背景值不参与训练，标签值设置为0。
            （2）对于一些多源数据模型的信息备注：MultiModelCNN适用信息为Multispectral、SAR、DEM；
                                                 HRN适用各类传感器信息。
            （3）使用库版本:certifi              2025.6.15
                            charset-normalizer   3.4.2
                            colorama             0.4.6
                            contourpy            1.3.0
                            cycler               0.12.1
                            einops               0.8.1
                            fightingcv_attention 1.0.0
                            filelock             3.18.0
                            fonttools            4.58.4
                            fsspec               2025.5.1
                            GDAL                 3.4.3
                            huggingface-hub      0.33.1
                            idna                 3.10
                            importlib_resources  6.5.2
                            Jinja2               3.1.6
                            joblib               1.5.1
                            kiwisolver           1.4.7
                            MarkupSafe           2.1.5
                            matplotlib           3.9.4
                            mpmath               1.3.0
                            networkx             3.2.1
                            numpy                1.26.4
                            packaging            25.0
                            pandas               2.3.0
                            pillow               11.2.1
                            pip                  25.1
                            protobuf             6.31.1
                            pyparsing            3.2.3
                            PyQt5                5.15.11
                            PyQt5-Qt5            5.15.2
                            PyQt5_sip            12.17.0
                            PySnooper            1.2.3
                            python-dateutil      2.9.0.post0
                            pytz                 2025.2
                            PyYAML               6.0.2
                            requests             2.32.4
                            safetensors          0.5.3
                            scikit-learn         1.6.1
                            scipy                1.13.1
                            setuptools           78.1.1
                            six                  1.17.0
                            sympy                1.14.0
                            tensorboardX         2.6.4
                            threadpoolctl        3.6.0
                            timm                 1.0.16
                            torch                2.8.0.dev20250621+cu128
                            torchaudio           2.8.0.dev20250622+cu128
                            TorchSnooper         0.8
                            torchsummary         1.5.1
                            torchvision          0.23.0.dev20250622+cu128
                            tqdm                 4.67.1
                            typing_extensions    4.14.0
                            tzdata               2025.2
                            urllib3              2.5.0
                            wheel                0.45.1
                            zipp                 3.23.0
            （4）CUDA版本：release 12.8, V12.8.61
                 cuDNN版本：8.9.7.29
