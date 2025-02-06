_base_ = ['./segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py']
channels = [16, 32, 64,128]
# checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b1_20220624-02e5a6a1.pth'  # noqa
custom_imports = dict(imports=['mmpre.models'], allow_failed_imports=False)
model = dict(
    backbone=dict(
        _delete_=True,
        type='mmpre.TIMMBackbone',  # 使用 mmcls 中的 timm 主干网络
        model_name='efficientvit_b0.r224_in1k',  # 使用 TIMM 中的 mobilevitv2_050
        features_only=True,
        pretrained=True,
        out_indices=(0,1, 2, 3)
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        # embed_dims=64
        ),
    decode_head=dict(in_channels=channels))
