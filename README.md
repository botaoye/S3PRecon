# S3PRecon

The official implementation for the **CVPR 2023** paper [_Self-Supervised Super-Plane for Neural 3D Reconstruction_](https://openaccess.thecvf.com/content/CVPR2023/papers/Ye_Self-Supervised_Super-Plane_for_Neural_3D_Reconstruction_CVPR_2023_paper.pdf).

## Installation and Setup
```bash
conda env create -f environment.yml
conda activate manhattan
```

## Usage

### Training
```python
python train_net.py --cfg_file configs/scannet/0084_self_plane.yaml gpus 0, exp_name scannet_0084_self_plane
```

### Evaluation
```python
python run.py --type evaluate --cfg_file configs/scannet/0084_self_plane.yaml gpus 0, exp_name scannet_0084_self_plane
```

### Mesh extraction
```python
python run.py --type mesh_extract --output_mesh result.obj --cfg_file configs/scannet/0084_self_plane.yaml gpus 0, exp_name scannet_0084_self_plane
```


## Acknowledgments
* Thanks for the [ManhattanSDF](https://github.com/zju3dv/manhattan_sdf), which helps us to quickly implement our ideas.
* [neurecon](https://github.com/ventusff/neurecon)
* [PlanarReconstruction](https://github.com/svip-lab/PlanarReconstruction)

## Citation
If our work is useful for your research, please consider citing:

```Bibtex
@inproceedings{ye2023s3p,
  title={Self-Supervised Super-Plane for Neural 3D Reconstruction},
  author={Ye, Botao and Liu, Sifei and Li, Xueting and Yang, Ming-Hsuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21415--21424},
  year={2023}
}
```