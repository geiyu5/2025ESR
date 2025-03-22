# Solution for (Team 08) Team mmSR: xxxxxxxxxxxxxxxxxx -- [NTIRE 2025 Challenge on Efficient Super-Resolution](https://cvlai.net/ntire/2025/)

<div align=center>
<img src="https://github.com/Amazingren/NTIRE2025_ESR/blob/main/figs/logo.png" width="400px"/> 
</div>

## Our result:
Valid PSNR : 26.96dB

Test PSNR : 27.05

FLOPs : 13.8547 [G]

Params : 0.2116 [M]

Average runtime of (valid) is : 12.624697 milliseconds on A6000

## The Environments

The evaluation environments adopted by us is recorded in the `requirements.txt`. After you built your own basic Python (Python = 3.9 in our setting) setup via either *virtual environment* or *anaconda*, please try to keep similar to it via:


## Install

- Step1: Git clone this repository
````  
git clone https://github.com/geiyu5/2025ESR.git
````
- Step2: install Pytorch compatible to your GPU (in this case, we follow the environment setting for NTIRE 2025 ESR):(python3.9)
````  
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
````
- Step3: install other libs via:
  ````
  pip install -r requirements.txt
  ````

The environment setting is kept as similar with [NTIRE2025 ESR](https://github.com/Amazingren/NTIRE2025_ESR)


## The Validation datasets
After downloaded all the necessary validate dataset ([DIV2K_LSDIR_valid_LR](https://drive.google.com/file/d/1YUDrjUSMhhdx1s-O0I1qPa_HjW-S34Yj/view?usp=sharing) and [DIV2K_LSDIR_valid_HR](https://drive.google.com/file/d/1z1UtfewPatuPVTeAAzeTjhEGk4dg2i8v/view?usp=sharing)), please organize them as follows:

```
|NTIRE2025_ESR_Challenge/
|--DIV2K_LSDIR_valid_HR/
|    |--000001.png
|    |--000002.png
|    |--...
|    |--000100.png
|    |--0801.png
|    |--0802.png
|    |--...
|    |--0900.png
|--DIV2K_LSDIR_valid_LR/
|    |--000001x4.png
|    |--000002x4.png
|    |--...
|    |--000100x4.png
|    |--0801x4.png
|    |--0802x4.png
|    |--...
|    |--0900.png
|--NTIRE2025_ESR/
|    |--...
|    |--test_demo.py
|    |--...
|--results/
|--......
```

## Running Validation
The shell script for validation is as follows: 
This shell script can be found in run.sh
```python
# --- Evaluation on LSDIR_DIV2K_valid datasets for One Method: ---
 CUDA_VISIBLE_DEVICES=0 python test_demo.py \
    --data_dir ./NTIRE2025_ESR_Challenge \
    --save_dir ./NTIRE2025_ESR/results \
    --ssim \
    --model_id 08
```

### Fix and Give the data_dir (HR & LR directory) and save_dir before running the command.

## Simply Run using this command
 ````
  sh run.sh
  ````


## References
If you feel this codebase and the report paper is useful for you, please cite our challenge report:
```
@inproceedings{ren2024ninth,
  title={The ninth NTIRE 2024 efficient super-resolution challenge report},
  author={Ren, Bin and Li, Yawei and Mehta, Nancy and Timofte, Radu and Yu, Hongyuan and Wan, Cheng and Hong, Yuxin and Han, Bingnan and Wu, Zhuoyuan and Zou, Yajun and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6595--6631},
  year={2024}
}
```


## License and Acknowledgement
This code repository is release under [MIT License](LICENSE). 
