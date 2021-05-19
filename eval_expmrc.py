# -*- coding: utf-8 -*-
from __future__ import print_function
import collections
import string
import re
import argparse
import json
import sys
import nltk

SP_CHARS = ['-',':','_','*','^','/','\\','~','`','+','=','，','。','：','？','！','“','”','；','’','《','》','……','·','、','「','」','（','）','－','～','『','』']

def mixed_segmentation(in_str):
	"""Compatible with English and Chinese (mixed)"""
	in_str = str(in_str).strip()
	segs_out = []
	temp_str = ""
	for ch in in_str:
		if re.search(r'[\u4e00-\u9fa5]', ch) or ch in SP_CHARS:
			if temp_str != "":
				segs_out.extend(nltk.word_tokenize(temp_str))
				temp_str = ""
			segs_out.append(ch)
		else:
			temp_str += ch

	# handling last part
	if temp_str != "":
		segs_out.extend(nltk.word_tokenize(temp_str))

	return segs_out

def normalize_answer(in_segs):
	"""Lower text and remove punctuation, articles and extra whitespace."""
	segs_out = []
	for seg in in_segs:
		if seg == 'a' or seg == 'an' or seg == 'the':
			continue
		if seg in set(string.punctuation + ''.join(SP_CHARS)):
			continue
		segs_out.append(seg.lower())
	return segs_out

def evaluate_span(ground_truth_file, prediction_file):
	answer_f1 = 0
	evidence_f1 = 0
	all_f1 = 0
	total_count = 0
	skip_count = 0
	for i in range(len(ground_truth_file["data"])):
		article = ground_truth_file["data"][i]
		for j in range(len(article['paragraphs'])):
			para = article['paragraphs'][j]
			context = para['context']

			for k in range(len(para['qas'])):
				qas = para['qas'][k]
				total_count += 1
				query_id    = qas['id']
				query_text  = qas['question']
				answers 	= [x['text'] for x in qas['answers']]
				evidences 	= qas['evidences']

				if query_id not in prediction_file:
					sys.stderr.write('Unanswered question: {}\n'.format(query_id))
					skip_count += 1
					continue

				answer_prediction = str(prediction_file[query_id]['answer'])
				evidence_prediction = str(prediction_file[query_id]['evidence'])

				temp_answer_f1 = calc_f1_score(answers, answer_prediction)
				temp_evidence_f1 = calc_f1_score(evidences, evidence_prediction)

				answer_f1 += temp_answer_f1
				evidence_f1 += temp_evidence_f1
				all_f1 += temp_answer_f1 * temp_evidence_f1

	all_f1_score = 100.0 * all_f1 / total_count
	answer_f1_score = 100.0 * answer_f1 / total_count
	evidence_f1_score = 100.0 * evidence_f1 / total_count
	return all_f1_score, answer_f1_score, evidence_f1_score, total_count, skip_count

def evaluate_multi_choice(ground_truth_file, prediction_file):
	answer_f1 = 0
	evidence_f1 = 0
	all_f1 = 0
	total_count = 0
	skip_count = 0
	for i in range(len(ground_truth_file["data"])):
		sample = ground_truth_file["data"][i]
		pid = sample['id']
		passage = sample['article']
		question = sample['questions']
		options = sample['options']
		answers = sample['answers']
		if 'evidences' in sample:
			evidences = sample['evidences']
		else:
			evidences = [''] * len(answers) 
		total_count += len(question)

		for j in range(len(answers)):
			pid_with_qid = pid + '-' + str(j)
			if pid_with_qid not in prediction_file:
				sys.stderr.write('Unanswered question: {}\n'.format(pid_with_qid))
				skip_count += 1
				continue

			answer_prediction = prediction_file[pid_with_qid]['answer']
			evidence_prediction = prediction_file[pid_with_qid]['evidence']

			temp_answer_f1 = 1 if answer_prediction==answers[j] else 0
			if 'evidences' in sample:
				temp_evidence_f1 = calc_f1_score(evidences[j], evidence_prediction)
			else:
				temp_evidence_f1 = 0

			answer_f1 += temp_answer_f1
			evidence_f1 += temp_evidence_f1
			all_f1 += temp_answer_f1 * temp_evidence_f1

	all_f1_score = 100.0 * all_f1 / total_count
	answer_f1_score = 100.0 * answer_f1 / total_count
	evidence_f1_score = 100.0 * evidence_f1 / total_count
	return all_f1_score, answer_f1_score, evidence_f1_score, total_count, skip_count

def calc_f1_score(answers, prediction):
	f1_scores = []
	pred_toks = normalize_answer(mixed_segmentation(prediction))
	for ans in answers:
		gold_toks = normalize_answer(mixed_segmentation(ans))
		common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
		num_same = sum(common.values())
		if len(gold_toks) == 0 or len(pred_toks) == 0:
			# If either is no-answer, then F1 is 1 if they agree, 0 otherwise
			f1_scores.append(int(gold_toks == pred_toks))
			continue
		if num_same == 0:
			f1_scores.append(0)
			continue
		precision = 1.0 * num_same / len(pred_toks)
		recall = 1.0 * num_same / len(gold_toks)
		f1 = (2 * precision * recall) / (precision + recall)
		f1_scores.append(f1)
	
	return max(f1_scores)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Evaluation Script for ExpMRC')
	parser.add_argument('dataset_file', help='Official dataset file')
	parser.add_argument('prediction_file', help='Your prediction File')
	args = parser.parse_args()

	ground_truth_file   = json.load(open(args.dataset_file, 'rb'))
	prediction_file     = json.load(open(args.prediction_file, 'rb'))

	data_version = ground_truth_file["version"]
	if 'squad' in data_version or 'cmrc2018' in data_version:
		all_f1, ans_f1, evi_f1, total_count, skip_count = evaluate_span(ground_truth_file, prediction_file)
	elif 'race' in data_version or 'c3' in data_version:
		all_f1, ans_f1, evi_f1, total_count, skip_count = evaluate_multi_choice(ground_truth_file, prediction_file)
	else:
		raise ValueError("Unsupported version: "+str(data_version))

	output_result = collections.OrderedDict()
	output_result['ALL_F1'] = '%.3f' % all_f1
	output_result['ANS_F1'] = '%.3f' % ans_f1
	output_result['EVI_F1'] = '%.3f' % evi_f1
	output_result['TOTAL'] = total_count
	output_result['SKIP'] = skip_count
	output_result['VERSION'] = data_version
	output_result['FILE'] = args.prediction_file
	print(json.dumps(output_result))
