# Sample Submission

Format for submission of each subset is identical. 
```json
{
  "KEY1": {"answer": "ANSWER_SPAN", "evidence": "EVIDENCE_TEXT"},
  "KEY2": {"answer": "ANSWER_SPAN", "evidence": "EVIDENCE_TEXT"},
  ...
}
```

1. For SQuAD and CMRC 2018, please format the `KEY` using `qid` of the question, which is the same with normal SQuAD/CMRC 2018 submission.
2. For RACE+ and C3, please format the `KEY` using `id` of the article and question number with `-` between them, such as `00052cc8-0`. Question number starts from `0`. **DO NOT SHUFFLE THE QUESTIONS!**

Please see each sample submission file on development set for details.
