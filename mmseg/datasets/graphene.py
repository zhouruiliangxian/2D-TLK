from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class graphene(BaseSegDataset):
    """LIP dataset.

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """

    METAINFO = dict(
        classes=('Background', 'monolayer', 'bilayer'),
        palette=(
            [[0, 0, 0],[0, 128, 0],[0, 0, 128]]
        ))
    # 指定图像扩展名、标注扩展名
    def __init__(self,
                 seg_map_suffix='.png',  # 标注mask图像的格式
                 reduce_zero_label=False,  # 类别ID为0的类别是否需要除去
                 **kwargs) -> None:
        super().__init__(
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
    # METAINFO = dict(
    #     classes=('monolayer', 'bilayer'),
    #     palette=(
    #         [[0, 128, 0], [0, 0, 128]]
    #     ))
    #
    # # 指定图像扩展名、标注扩展名
    # def __init__(self,
    #              seg_map_suffix='.png',  # 标注mask图像的格式
    #              reduce_zero_label=True,  # 类别ID为0的类别是否需要除去
    #              **kwargs) -> None:
    #     super().__init__(
    #         seg_map_suffix=seg_map_suffix,
    #         reduce_zero_label=reduce_zero_label,
    #         **kwargs)

