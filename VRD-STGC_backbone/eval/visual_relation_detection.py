from collections import defaultdict, OrderedDict
import numpy as np
import torch
from .common import voc_ap, viou

def get_label_ids():
    pred2pid = dict()
    pred = set()
    for i, line in enumerate(open("../data/relation.txt")):
        label= line.strip().split()[0]     
        pred.add(label)
    pred = sorted(pred)
    for i, name in enumerate(pred):
        pred2pid[name] = i
    return pred2pid
def eval_detection_scores_mean(gt_relations, pred_relations, viou_threshold, allow_misalign=False):
    """
    allow_misalign: allow temporal misalignment between subject and object in the prediction;
                    this require 'duration' being replaced by 'sub_duration' and 'obj_duration'
    """
    pred_relations = sorted(pred_relations, key=lambda x: x['score'], reverse=True)
    gt_detected = np.zeros((len(gt_relations),), dtype=bool)
    hit_scores = np.ones((max(len(pred_relations), len(gt_relations)),)) * -np.inf
    hit_label = np.ones((max(len(pred_relations), len(gt_relations)),)) * -np.inf
    label_ids = get_label_ids()
    for pred_idx, pred_relation in enumerate(pred_relations):
        ov_max = -float('Inf')
        k_max = -1
        for gt_idx, gt_relation in enumerate(gt_relations):
            if not gt_detected[gt_idx] and tuple(pred_relation['triplet']) == tuple(gt_relation['triplet']):
                if allow_misalign and 'sub_duration' in pred_relation:
                    sub_duration = pred_relation['sub_duration']
                    obj_duration = pred_relation['obj_duration']
                else:
                    sub_duration = pred_relation['duration']
                    obj_duration = pred_relation['duration']
                    
                s_iou = viou(pred_relation['sub_traj'], sub_duration,
                        gt_relation['sub_traj'], gt_relation['duration'])
                o_iou = viou(pred_relation['obj_traj'], obj_duration,
                        gt_relation['obj_traj'], gt_relation['duration'])
                ov = min(s_iou, o_iou)

                if ov >= viou_threshold and ov > ov_max:
                    ov_max = ov
                    k_max = gt_idx
        if k_max >= 0:
            hit_scores[pred_idx] = pred_relation['score']
            hit_label[pred_idx]=(label_ids[str(pred_relation["triplet"][1])])
            gt_detected[k_max] = True
    tp = np.isfinite(hit_scores)
    fp = ~tp
    cum_tp = np.cumsum(tp).astype(np.float32)
    cum_fp = np.cumsum(fp).astype(np.float32)
    rec = cum_tp / np.maximum(len(gt_relations), np.finfo(np.float32).eps)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float32).eps)
    return prec, rec, hit_scores, hit_label

def eval_detection_scores_mean_predcls(gt_relations, pred_relations, viou_threshold, allow_misalign=False):
    """
    allow_misalign: allow temporal misalignment between subject and object in the prediction;
                    this require 'duration' being replaced by 'sub_duration' and 'obj_duration'
    """
    pred_relations = sorted(pred_relations, key=lambda x: x['score'], reverse=True)
    gt_detected = np.zeros((len(gt_relations),), dtype=bool)
    hit_scores = np.ones((max(len(pred_relations), len(gt_relations)),)) * -np.inf
    hit_label = np.ones((max(len(pred_relations), len(gt_relations)),)) * -np.inf
    label_ids = get_label_ids()
    for pred_idx, pred_relation in enumerate(pred_relations):
        ov_max = -float('Inf')
        k_max = -1
        for gt_idx, gt_relation in enumerate(gt_relations):
            if not gt_detected[gt_idx] and tuple(pred_relation['triplet'][1]) == tuple(gt_relation['triplet'][1]):
                if allow_misalign and 'sub_duration' in pred_relation:
                    sub_duration = pred_relation['sub_duration']
                    obj_duration = pred_relation['obj_duration']
                else:
                    sub_duration = pred_relation['duration']
                    obj_duration = pred_relation['duration']
                    
                s_iou = 1
                o_iou = 1
                ov = min(s_iou, o_iou)

                if ov >= viou_threshold and ov > ov_max:
                    ov_max = ov
                    k_max = gt_idx
        if k_max >= 0:
            hit_scores[pred_idx] = pred_relation['score']
            hit_label[pred_idx]=(label_ids[str(pred_relation["triplet"][1])])
            gt_detected[k_max] = True
    tp = np.isfinite(hit_scores)
    fp = ~tp
    cum_tp = np.cumsum(tp).astype(np.float32)
    cum_fp = np.cumsum(fp).astype(np.float32)
    rec = cum_tp / np.maximum(len(gt_relations), np.finfo(np.float32).eps)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float32).eps)
    return prec, rec, hit_scores, hit_label

def eval_detection_scores(gt_relations, pred_relations, viou_threshold, allow_misalign=False):
    """
    allow_misalign: allow temporal misalignment between subject and object in the prediction;
                    this require 'duration' being replaced by 'sub_duration' and 'obj_duration'
    """
    pred_relations = sorted(pred_relations, key=lambda x: x['score'], reverse=True)
    gt_detected = np.zeros((len(gt_relations),), dtype=bool)
    hit_scores = np.ones((max(len(pred_relations), len(gt_relations)),)) * -np.inf
    for pred_idx, pred_relation in enumerate(pred_relations):
        ov_max = -float('Inf')
        k_max = -1
        for gt_idx, gt_relation in enumerate(gt_relations):
            if not gt_detected[gt_idx] and tuple(pred_relation['triplet']) == tuple(gt_relation['triplet']):
                if allow_misalign and 'sub_duration' in pred_relation:
                    sub_duration = pred_relation['sub_duration']
                    obj_duration = pred_relation['obj_duration']
                else:
                    sub_duration = pred_relation['duration']
                    obj_duration = pred_relation['duration']
                    
                s_iou = viou(pred_relation['sub_traj'], sub_duration,
                        gt_relation['sub_traj'], gt_relation['duration'])
                o_iou = viou(pred_relation['obj_traj'], obj_duration,
                        gt_relation['obj_traj'], gt_relation['duration'])
                ov = min(s_iou, o_iou)

                if ov >= viou_threshold and ov > ov_max:
                    ov_max = ov
                    k_max = gt_idx
        if k_max >= 0:
            hit_scores[pred_idx] = pred_relation['score']
            gt_detected[k_max] = True
    tp = np.isfinite(hit_scores)
    fp = ~tp
    cum_tp = np.cumsum(tp).astype(np.float32)
    cum_fp = np.cumsum(fp).astype(np.float32)
    rec = cum_tp / np.maximum(len(gt_relations), np.finfo(np.float32).eps)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float32).eps)
    return prec, rec, hit_scores


def eval_tagging_scores(gt_relations, pred_relations, min_pred_num=0):
    pred_relations = sorted(pred_relations, key=lambda x: x['score'], reverse=True)
    # ignore trajectories
    gt_triplets = set(tuple(r['triplet']) for r in gt_relations)
    pred_triplets = []
    hit_scores = []
    for r in pred_relations:
        triplet = tuple(r['triplet'])
        if not triplet in pred_triplets:
            pred_triplets.append(triplet)
            hit_scores.append(r['score'])
    hit_scores.extend([-np.inf]*(min_pred_num-len(hit_scores)))
    hit_scores = np.asarray(hit_scores)
    for i, t in enumerate(pred_triplets):
        if not t in gt_triplets:
            hit_scores[i] = -np.inf
    tp = np.isfinite(hit_scores)
    fp = ~tp
    cum_tp = np.cumsum(tp).astype(np.float32)
    cum_fp = np.cumsum(fp).astype(np.float32)
    rec = cum_tp / np.maximum(len(gt_triplets), np.finfo(np.float32).eps)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float32).eps)
    return prec, rec, hit_scores


def evaluate_predcls(groundtruth, prediction, viou_threshold=0.5,
            det_nreturns=[50, 100], tag_nreturns=[1, 5, 10],
            allow_misalign=False, verbose=True):
    """ evaluate visual relation detection and visual 
    relation tagging.
    """
    if allow_misalign:
        print('[warning] subject and object misalignment allowed (non-official support)')
    video_ap = dict()
    tot_scores = defaultdict(list)
    tot_tp = defaultdict(list)
    prec_at_n = defaultdict(list)
    tot_gt_relations = 0

    recall = [defaultdict(list) for n in range(132)]
    label_ids = get_label_ids()

    if verbose:
        print('[info] computing metric scores over {} videos...'.format(len(groundtruth)))
    for vid, gt_relations in groundtruth.items():
        if len(gt_relations)==0:
            continue
        tot_gt_relations += len(gt_relations)
        predict_relations = prediction.get(vid, [])

        gt_num = [0 for i in range(132)]
        for gr in gt_relations:
            index = label_ids[str(gr["triplet"][1])]
            gt_num[index] = gt_num[index]+1
        
        # compute average precision and recalls in detection setting
        det_prec, det_rec, det_scores, det_label = eval_detection_scores_mean_predcls(
                gt_relations, predict_relations, viou_threshold, allow_misalign=allow_misalign)
        video_ap[vid] = voc_ap(det_rec, det_prec)
        tp = np.isfinite(det_scores)
        for nre in det_nreturns:
            cut_off = min(nre, det_scores.size)
            tmp_det_score = det_scores[:cut_off]
            tmp_det_label = det_label[:cut_off]
            det_score_iso = [[]for i in range(132)]
            for i in range(cut_off):
                index = tmp_det_label[i]
                score = tmp_det_score[i]
                if np.isfinite(index)==True:
                    index = int(index)
                    det_score_iso[index].append(score)
            for i in range(132):
                tps_iso = np.isfinite(det_score_iso[i])
                sort_indices = np.argsort(det_score_iso[i])[::-1]#argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y 使用[::-1],可以建立X从大到小的索引
                tps_iso = tps_iso[sort_indices]
                cum_tp = np.cumsum(tps_iso).astype(np.float32)
                if (gt_num[i]!=0):
                    if (len(tps_iso)==0):
                        recall[i][nre].append(0.)
                    else:
                        recall[i][nre].append((cum_tp/gt_num[i])[-1])
        for nre in det_nreturns:
            cut_off = min(nre, det_scores.size)
            tot_scores[nre].append(det_scores[:cut_off])
            tot_tp[nre].append(tp[:cut_off])
        # compute precisions in tagging setting
        tag_prec, _, _ = eval_tagging_scores(gt_relations, predict_relations, max(tag_nreturns))
        for nre in tag_nreturns:
            prec_at_n[nre].append(tag_prec[nre-1])
    
    output = OrderedDict()
    # calculate mean ap for detection
    output['detection mean AP'] = np.mean(list(video_ap.values()))
    # calculate recall for detection
    for nre in det_nreturns:
        scores = np.concatenate(tot_scores[nre])
        tps = np.concatenate(tot_tp[nre])
        sort_indices = np.argsort(scores)[::-1]
        tps = tps[sort_indices]
        cum_tp = np.cumsum(tps).astype(np.float32)
        rec = cum_tp / np.maximum(tot_gt_relations, np.finfo(np.float32).eps)
        output['detection recall@{}'.format(nre)] = rec[-1]
    # calculate mean precision for tagging
    for nre in tag_nreturns:
        output['tagging precision@{}'.format(nre)] = np.mean(prec_at_n[nre])
    tail_list = [0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 21, 22, 23, 24, 25, 27, 28, 29, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 48, 50, 52, 56, 57, 58, 66, 69, 71, 74, 75, 77, 80, 81, 83, 87, 89, 95, 97, 102, 104, 109, 110, 111, 112, 113, 114, 115, 116, 119, 120, 123]
    body_list = [2, 5, 18, 20, 26, 32, 37, 49, 51, 53, 54, 55, 63, 64, 67, 68, 73, 76, 78, 79, 82, 84, 86, 88, 90, 91, 92, 93, 101, 105, 107, 108, 118, 121, 127, 129]
    head_list =  [30, 31, 46, 47, 59, 60, 61, 62, 65, 70, 72, 85, 94, 96, 98, 99, 100, 103, 106, 117, 122, 124, 125, 126, 128, 130, 131]
    for nre in det_nreturns:
        mean_recall = [0.0 for n in range(132)]
        for i in range(132):
            if (len(recall[i])==0):
                mean_recall[i] = 0
            else:
                #print(recall[i][nre])
                #print(len(recall[i][nre]))
                mean_recall[i] =  np.mean(list(recall[i][nre]))
        output['detection mean_recall_list@{}'.format(nre)] = mean_recall
        tail = torch.tensor(mean_recall)[tail_list]
        head = torch.tensor(mean_recall)[head_list]
        body = torch.tensor(mean_recall)[body_list]
        output['detection mean_recall@{}'.format(nre)] = np.mean(list(mean_recall))
        output['detection mean_headrecall@{}'.format(nre)] = np.mean(list(head))
        output['detection mean_tailrecall@{}'.format(nre)] = np.mean(list(tail))
        output['detection mean_bodyrecall@{}'.format(nre)] = np.mean(list(body))
    return output


def evaluate(groundtruth, prediction, viou_threshold=0.5,
            det_nreturns=[50, 100], tag_nreturns=[1, 5, 10],
            allow_misalign=False, verbose=True):
    """ evaluate visual relation detection and visual 
    relation tagging.
    """
    if allow_misalign:
        print('[warning] subject and object misalignment allowed (non-official support)')
    video_ap = dict()
    tot_scores = defaultdict(list)
    tot_tp = defaultdict(list)
    prec_at_n = defaultdict(list)
    tot_gt_relations = 0

    recall = [defaultdict(list) for n in range(132)]
    label_ids = get_label_ids()

    if verbose:
        print('[info] computing metric scores over {} videos...'.format(len(groundtruth)))
    for vid, gt_relations in groundtruth.items():
        if len(gt_relations)==0:
            continue
        tot_gt_relations += len(gt_relations)
        predict_relations = prediction.get(vid, [])

        gt_num = [0 for i in range(132)]
        for gr in gt_relations:
            index = label_ids[str(gr["triplet"][1])]
            gt_num[index] = gt_num[index]+1
        
        # compute average precision and recalls in detection setting
        det_prec, det_rec, det_scores, det_label = eval_detection_scores_mean(
                gt_relations, predict_relations, viou_threshold, allow_misalign=allow_misalign)
        video_ap[vid] = voc_ap(det_rec, det_prec)
        tp = np.isfinite(det_scores)
        for nre in det_nreturns:
            cut_off = min(nre, det_scores.size)
            tmp_det_score = det_scores[:cut_off]
            tmp_det_label = det_label[:cut_off]
            det_score_iso = [[]for i in range(132)]
            for i in range(cut_off):
                index = tmp_det_label[i]
                score = tmp_det_score[i]
                if np.isfinite(index)==True:
                    index = int(index)
                    det_score_iso[index].append(score)
            for i in range(132):
                tps_iso = np.isfinite(det_score_iso[i])
                sort_indices = np.argsort(det_score_iso[i])[::-1]#argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y 使用[::-1],可以建立X从大到小的索引
                tps_iso = tps_iso[sort_indices]
                cum_tp = np.cumsum(tps_iso).astype(np.float32)
                if (gt_num[i]!=0):
                    if (len(tps_iso)==0):
                        recall[i][nre].append(0.)
                    else:
                        recall[i][nre].append((cum_tp/gt_num[i])[-1])
        for nre in det_nreturns:
            cut_off = min(nre, det_scores.size)
            tot_scores[nre].append(det_scores[:cut_off])
            tot_tp[nre].append(tp[:cut_off])
        # compute precisions in tagging setting
        tag_prec, _, _ = eval_tagging_scores(gt_relations, predict_relations, max(tag_nreturns))
        for nre in tag_nreturns:
            prec_at_n[nre].append(tag_prec[nre-1])
    
    output = OrderedDict()
    # calculate mean ap for detection
    output['detection mean AP'] = np.mean(list(video_ap.values()))
    # calculate recall for detection
    for nre in det_nreturns:
        scores = np.concatenate(tot_scores[nre])
        tps = np.concatenate(tot_tp[nre])
        sort_indices = np.argsort(scores)[::-1]
        tps = tps[sort_indices]
        cum_tp = np.cumsum(tps).astype(np.float32)
        rec = cum_tp / np.maximum(tot_gt_relations, np.finfo(np.float32).eps)
        output['detection recall@{}'.format(nre)] = rec[-1]
    # calculate mean precision for tagging
    for nre in tag_nreturns:
        output['tagging precision@{}'.format(nre)] = np.mean(prec_at_n[nre])   
    tail_list = [0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 21, 22, 23, 24, 25, 27, 28, 29, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 48, 50, 52, 56, 57, 58, 66, 69, 71, 74, 75, 77, 80, 81, 83, 87, 89, 95, 97, 102, 104, 109, 110, 111, 112, 113, 114, 115, 116, 119, 120, 123]
    body_list = [2, 5, 18, 20, 26, 32, 37, 49, 51, 53, 54, 55, 63, 64, 67, 68, 73, 76, 78, 79, 82, 84, 86, 88, 90, 91, 92, 93, 101, 105, 107, 108, 118, 121, 127, 129]
    head_list =  [30, 31, 46, 47, 59, 60, 61, 62, 65, 70, 72, 85, 94, 96, 98, 99, 100, 103, 106, 117, 122, 124, 125, 126, 128, 130, 131]
    for nre in det_nreturns:
        mean_recall = [0.0 for n in range(132)]
        for i in range(132):
            if (len(recall[i])==0):
                mean_recall[i] = 0
            else:
                mean_recall[i] =  np.mean(list(recall[i][nre]))
        output['detection mean_recall_list@{}'.format(nre)] = mean_recall
        output['detection mean_recall@{}'.format(nre)] = np.mean(list(mean_recall))
        tail = torch.tensor(mean_recall)[tail_list]
        head = torch.tensor(mean_recall)[head_list]
        body = torch.tensor(mean_recall)[body_list]
        output['detection mean_recall@{}'.format(nre)] = np.mean(list(mean_recall))
        output['detection mean_headrecall@{}'.format(nre)] = np.mean(list(head))
        output['detection mean_tailrecall@{}'.format(nre)] = np.mean(list(tail))
        output['detection mean_bodyrecall@{}'.format(nre)] = np.mean(list(body))
    return output


def print_scores(scores, score_variance=None):
    for setting in scores.keys():
        print('[setting] {}'.format(setting))
        for metric in scores[setting].keys():
            if score_variance is not None:
                print('\t{}:\t{:.4f} \u00B1 {:.4f}'.format(metric, scores[setting][metric], score_variance[setting][metric]))
            else:
                if metric == "detection mean_recall_list@50" or metric == "detection mean_recall_list@100":
                    print('\t{}:\t{}'.format(metric, scores[setting][metric]))
                else:
                    print('\t{}:\t{:.4f}'.format(metric, scores[setting][metric]))


if __name__ == "__main__":
    """
    You can directly run this script from the parent directory, e.g.,
    python -m evaluation.visual_relation_detection val_relation_groundtruth.json val_relation_prediction.json
    """
    import json
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Video visual relation detection evaluation.')
    parser.add_argument('groundtruth', type=str,help='A ground truth JSON file generated by yourself')
    parser.add_argument('prediction', type=str, help='A prediction file')
    args = parser.parse_args()
    
    print('[info] loading ground truth from {}'.format(args.groundtruth))
    with open(args.groundtruth, 'r') as fp:
        gt = json.load(fp)
    print('[info] number of videos in ground truth: {}'.format(len(gt)))

    print('[info] loading prediction from {}'.format(args.prediction))
    with open(args.prediction, 'r') as fp:
        pred = json.load(fp)
    print('[info] number of videos in prediction: {}'.format(len(pred['results'])))

    mean_ap, rec_at_n, mprec_at_n = evaluate(gt, pred['results'])
