# ExpMRC

With the development of the pre-trained language models (PLMs), achieving human-level performance on several machine reading comprehension (MRC) dataset is not as hard as it used to be. However, the explainability behind these artifacts still remains unclear, raising concerns on utilizing these models in real-life applications. To improve the explainability of MRC tasks, we propose ExpMRC benchmark. 

**ExpMRC** is a benchmark for **Exp**lainability Evaluation of **M**achine **R**eading **C**omprehension. ExpMRC contains four subsets of popular MRC datasets with additionally annotated evidences, including [SQuAD](https://www.aclweb.org/anthology/D16-1264/), [CMRC 2018](https://www.aclweb.org/anthology/D19-1600/), RACE+ (similar to [RACE](https://www.aclweb.org/anthology/D17-1082/)), and [C3](https://www.aclweb.org/anthology/2020.tacl-1.10/), covering span-extraction and multiple-choice questions MRC tasks in both English and Chinese. 

To achieve a higher score in ExpMRC, the model should not only give a correct answer for the question but also give a passage span as the evidence text. We greatly welcome the submission that could be generalized well on different languages and types of MRC tasks with *unsupervised* or *semi-supervised* approaches.

**ExpMRC: Explainability Evaluation for Machine Reading Comprehension**  
*Yiming Cui, Ting Liu, Wanxiang Che, Zhigang Chen, Shijin Wang*

arXiv pre-print: [https://arxiv.org/abs/2105.04126](https://arxiv.org/abs/2105.04126)

Leaderboard: [https://ymcui.github.io/expmrc/](https://ymcui.github.io/expmrc/)

## News

Thank you for your interest in our dataset. We are about to release the dataset and baseline codes in the next few weeks. Stay tuned!


## Submission to Leaderboard (Not Ready Yet)

Please visit our leaderboard for more information: [https://ymcui.github.io/expmrc/](https://ymcui.github.io/expmrc/)

To preserve the integrity of test results and improve the reproducibility, **we do not release the test sets to the public**. Instead, we require you to upload your model onto CodaLab, so that we can run it on the test sets for you. You can follow the instructions on CodaLab (which is similar to SQuAD, CMRC 2018 submission).

You can submit your model on one or more subsets in ExpMRC. 

Sample submission files are shown in `sample_submission` folder.


## Citation

If you are using our benchmark in your work, please cite:

```
@article{cui2021expmrc,
  title={ExpMRC: Explainability Evaluation for Machine Reading Comprehension},
  author={Cui, Yiming and Liu, Ting and Che, Wanxiang and Chen, Zhigang and Wang, Shijin},
  journal={arXiv preprint arXiv:2105.04126},
  year={2021}
}
```



## Contact us

Please submit an issue.