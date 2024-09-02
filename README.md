# Assist Non-native Viewers: Multimodal Cross-Lingual Summarization for How2 Videos
The original conference version was accepted by *EMNLP 2022*, and the extended journal version has been accepted by *TPAMI*.

## Data Preparing
The reorganized How2-MCLS text data can be downloaded from here [[Baidu Netdisk, Passcode: a9df]](https://pan.baidu.com/s/1sHZfz_ACejInd7B0ON4Ibw?pwd=a9df), as well as video features [[Baidu Netdisk, Passcode: eqqj]](https://pan.baidu.com/s/1JxkceABDIDkkP3SS3ejX5g) (derived from the original How2 dataset). The original How2 dataset for multimodal summarization is provided by [https://github.com/srvk/how2-dataset](https://github.com/srvk/how2-dataset).

## Preprocessing
Some demo data is placed in "data/demo_data" folder, and you can replace the demo data with the full How2-MCLS dataset, following the format of "data/demo_data" folder. Then run the following command to preprocess the data. This code takes the Pt2En scenario as an example for demonstration.

 ```python
python preprocess.py #Please modify the data storage path configuration.
 ```

## Training and Prediction
After data preprocessing, you can run the following script commands to execute the training and prediction procedures of the proposed models.

VDF

```
bash run_scripts/VDF.sh
```

VDF-TS-E

```
bash run_scripts/VDF-TS-E.sh
```

VDF-TS-V

```
bash run_scripts/VDF-TS-V.sh
```

VDF-TS-E2, using language-adaptive warping distillation (LAWD) to replace adaptive pooling distillation.

```
bash run_scripts/VDF-TS-E2.sh
```

VDF-TS-V2, using LAWD to replace adaptive pooling distillation.

```
bash run_scripts/VDF-TS-V2.sh
```


## Evaluation
[nmtpytorch](https://github.com/srvk/how2-dataset) library is used to evaluate models, which includes BLEU (1, 2, 3, 4), ROUGE-L, METEOR, and CIDEr evaluation metrics. 

As an alternative, [nlg-eval](https://github.com/Maluuba/nlg-eval) evaluation library can obtain the same evaluation scores as nmtpytorch.

In addition, [ROUGE](https://github.com/neural-dialogue-metrics/rouge) evaluation library is used to calculate the ROUGE (1, 2, L) score.

## Acknowledgement
We are very grateful that the code is based on [MFN](https://github.com/forkarinda/MFN), [nmtpytorch](https://github.com/srvk/how2-dataset), [fairseq](https://github.com/pytorch/fairseq), [machine-translation](https://github.com/tangbinh/machine-translation), [pytorch-softdtw-cuda](https://github.com/Maghoumi/pytorch-softdtw-cuda), and [Transformers](https://github.com/huggingface/transformers).

## Citation
```
@inproceedings{liu2022assist,
  title={Assist non-native viewers: Multimodal cross-lingual summarization for how2 videos},
  author={Liu, Nayu and Wei, Kaiwen and Sun, Xian and Yu, Hongfeng and Yao, Fanglong and Jin, Li and Zhi, Guo and Xu, Guangluan},
  booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
  pages={6959--6969},
  year={2022}
}
@article{liu2024multimodal,
  title={Multimodal Cross-lingual Summarization for Videos: A Revisit in Knowledge Distillation Induced Triple-stage Training Method},
  author={Liu, Nayu and Wei, Kaiwen and Yang, Yong and Tao, Jianhua and Sun, Xian and Yao, Fanglong and Yu, Hongfeng and Jin, Li and Lv, Zhao and Fan, Cunhang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  note = {Early Access},
  publisher={IEEE}
}
```
