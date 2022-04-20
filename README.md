<p align="center">
    <br>
    <img src="./banner.png" width="500"/>
    <br>
</p>
<p align="center">
    <a href="https://github.com/ymcui/expmrc/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/ymcui/expmrc.svg?color=blue&style=flat-square">
    </a>
</p>

With the development of the pre-trained language models (PLMs), achieving human-level performance on several machine reading comprehension (MRC) dataset is not as hard as it used to be. However, the explainability behind these artifacts still remains unclear, raising concerns on utilizing these models in real-life applications. To improve the explainability of MRC tasks, we propose ExpMRC benchmark. 

**ExpMRC** is a benchmark for **Exp**lainability Evaluation of **M**achine **R**eading **C**omprehension. ExpMRC contains four subsets of popular MRC datasets with additionally annotated evidences, including [SQuAD](https://www.aclweb.org/anthology/D16-1264/), [CMRC 2018](https://www.aclweb.org/anthology/D19-1600/), RACE<sup>+</sup> (similar to [RACE](https://www.aclweb.org/anthology/D17-1082/)), and [C<sup>3</sup>](https://www.aclweb.org/anthology/2020.tacl-1.10/), covering span-extraction and multiple-choice questions MRC tasks in both English and Chinese. 

To achieve a higher score in ExpMRC, the model should not only give a correct answer for the question but also give a passage span as the evidence text. We greatly welcome the submission that could be generalized well on different languages and types of MRC tasks with *unsupervised* or *semi-supervised* approaches.

**ExpMRC: Explainability Evaluation for Machine Reading Comprehension**

- [Yiming Cui](https://ymcui.com), Ting Liu, Wanxiang Che, Zhigang Chen, Shijin Wang
- Published in [Heliyon](https://www.cell.com/heliyon)

[[Official Publication]](https://www.cell.com/heliyon/fulltext/S2405-8440(22)00578-3) [[arXiv pre-print]](https://arxiv.org/abs/2105.04126) [[**Leaderboard**]](https://ymcui.github.io/expmrc/) [[Papers With Code]](https://paperswithcode.com/dataset/expmrc) 

## News

[Apri 19, 2022] **Our paper is officially published in [Heliyon](https://www.cell.com/heliyon).**

[June 22, 2021] Baseline codes and pseudo training data are available, check `baseline` and `pseudo-training-data` directory.

[May 24, 2021] We have released our dataset, check `data` directory. The submission site is also open.

[May 17, 2021] Thank you for your interest in our dataset. We are about to release the dataset and baseline codes in the next few weeks, hopefully on late May. Stay tuned!


## Directory Guide
```
| -- **ExpMRC Root**  
    | -- baseline               # Baseline codes  
    | -- data                   # ExpMRC development sets  
    | -- pseudo_training_data   # Pseudo training data  
    | -- sample_submission      # Sample submission files for ExpMRC  
    | -- eval_expmrc.py         # Evaluation script  
```

As stated in the paper, we **DO NOT** provide any training data. We intend to encourage our community to develop unsupervised or semi-supervised approaches for promoting Explainable MRC. Nonetheless, we provide the pseudo training data that was used in our paper. Please check `pseudo-training-data` directory.


## Submission to Leaderboard

Please visit our leaderboard for more information: [https://ymcui.github.io/expmrc/](https://ymcui.github.io/expmrc/)

To preserve the integrity of test results and improve the reproducibility, **we do not release the test sets to the public**. Instead, we require you to upload your model onto CodaLab, so that we can run it on the test sets for you. You can follow the instructions on CodaLab (which is similar to SQuAD, CMRC 2018 submission). You can submit your model on one or more subsets in ExpMRC. Sample submission files are shown in `sample_submission` directory.

Submission policies:
1. You are free to use any open-source MRC data or automatically generated data for training your systems (both labeled and unlabeled).
2. You are **NOT** allowed to use any **publicly unavailable** human-annotated data for training.
3. We do not encourage using the development set of ExpMRC for training (though it is not prohibited). You should declare whether the system is trained by using the whole/part of the development set. Such submissions will be marked with an asterisk (*).


## Citation

If you are using our benchmark in your work, please cite:

```
@article{cui-etal-2022-expmrc,
  title={ExpMRC: Explainability Evaluation for Machine Reading Comprehension},
  author={Cui, Yiming and Liu, Ting and Che, Wanxiang and Chen, Zhigang and Wang, Shijin},
  journal={Heliyon},
  year={2022},
  volume={8},
  issue={4},
  pages={e09290},
  issn={2405-8440},
  doi={https://doi.org/10.1016/j.heliyon.2022.e09290}
}
```


## Acknowledgment

[Yiming Cui](https://ymcui.com) would like to thank [Google TPU Research Cloud (TRC)](https://g.co/tfrc) program for providing computing resource.
We also thank [SQuAD team](https://rajpurkar.github.io/SQuAD-explorer/) for open-sourcing their website template.


## Contact us

Please submit an issue.