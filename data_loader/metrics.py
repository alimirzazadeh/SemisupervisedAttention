'''
Functions to compute metrics.

The main function to use is compute_metrics.

If you want to compute metrics one batch at a time, use StreamMetrics.

Example:
    
    Here is some example usage of compute_metrics:
    
        Y_true = ...
        Y_preds = ...

        metrics = compute_metrics(Y_true, Y_preds, ['acc', 'pre', 'rec', 'f1'])
    
    Then the metrics variable will be of the form:
    [('acc', ...), ('pre', ...), ('rec', ...), ('f1', ...)]
    
    Here is some example usage of StreamMetrics:
        
        streaming_metrics = StreamMetrics(['acc', 'pre', 'rec', 'f1'])
        
        for batch in dataloader:
            Y_true = batch[...]
            Y_preds = model(batch[...])
            
            streaming_metrics.update(Y_true, Y_preds)
            
        metrics = streaming_metrics.compute()
        
    Then the metrics variable will be of the form:
    [('acc', ...), ('pre', ...), ('rec', ...), ('f1', ...)]
'''

import numpy as np
import torch
from typing import Optional, Mapping, Dict, Sequence, Tuple, List, Union

NpOrTensor = Union[np.ndarray, torch.tensor]

np_arr = np.zeros(1)
torch_arr = torch.tensor([0])

class StreamMetrics:
    def __init__(
        self,
        metrics: Optional[Sequence[str]] = ['acc', 'pre', 'rec', 'f1'],
        pos_class: Optional[int] = 1,
        abstain_val: Optional[int] = None,
        num_classes: Optional[int] = 1,
    ) -> None:
        '''
        Initialize the state to keep track of for streaming metrics.
        
        Args:
            metrics (Sequence[str]): List of metrics to compute. Currently,
                'acc', 'pre', 'rec', 'f1', 'tp', 'tn', 'fp', and 'fn' are
                supported. acc, pre, rec, and f1 are computed by default.
                acc is top-1 accuracy.
                pre, rec, and f1 are precision, recall, and f1 for a binary
                    classifier. These metrics assume that 1 is the label of
                    the positive class.
                tp, tn, fp, and fn are true positive, true negative, false
                    positive, and false negative
        '''
        self.metrics = metrics
        self.total = 0
        self.num_classes = num_classes
        if 'acc' in metrics:
            self.num_correct = 0
        if ('pre' in metrics or 'rec' in metrics or 'f1' in metrics or
            'tp' in metrics or 'tn' in metrics or 'fp' in metrics or 
            'fn' in metrics):
            self.compute_tp_tn_fp_fn_total_pos = True
            if num_classes == 1:
                self.tp = 0
                self.tn = 0
                self.fp = 0
                self.fn = 0
                self.total_pos = 0
            else:
                self.classes = []
                self.tp = {}
                self.tn = {}
                self.fp = {}
                self.fn = {}
                self.total_pos = {}
        else:
            self.compute_tp_tn_fp_fn_total_pos = False
        self.using_np = None
        self.pos_class = pos_class
        self.abstain_val = abstain_val
    
    def update(
        self,
        Y_true: NpOrTensor,
        Y_preds: NpOrTensor,
    ) -> None:
        '''
        Update the running state with labels and predictions from a batch.
        
        Args:
            Y_true (Numpy array/PyTorch tensor): Ground-truth labels.
                Can either be a Numpy array or a PyTorch tensor.
            Y_preds (Numpy array/PyTorch tensor): Predictions.
                Can be a Numpy array or a PyTorch tensor, but needs to
                have the same type as Y_true.
                If a PyTorch tensor, needs to be on the same device (GPU
                or DRAM) as Y_true.
        '''

        if self.using_np is None:
            self.using_np = _is_using_np(Y_true, Y_preds)
        else:
            if self.using_np != _is_using_np(Y_true, Y_preds):
                raise NotImplementedError(
                    'Switch between Numpy arrays and PyTorch tensors unsupported'
                )
        
        if self.abstain_val is None:
            self.total += Y_true.shape[0]
        else:
            batch_total = (Y_preds != self.abstain_val).sum()
            if not self.using_np:
                batch_total = batch_total.itme()
            self.total += batch_total
        if 'acc' in self.metrics:
            self.num_correct += _compute_corrects(Y_true, Y_preds, self.using_np,
                                                 self.abstain_val)
        if self.compute_tp_tn_fp_fn_total_pos:
            if self.num_classes == 1:
                (batch_tp, batch_tn, batch_fp,
                 batch_fn, batch_total_pos) = _compute_tp_tn_fp_fn_total_pos(
                    Y_true, Y_preds, self.using_np, self.pos_class,
                    self.abstain_val
                )
                self.tp += batch_tp
                self.tn += batch_tn
                self.fp += batch_fp
                self.fn += batch_fn
                self.total_pos += batch_total_pos
            else:
                classes_true = set([int(i) for i in Y_true])
                classes_pred = set([int(i) for i in Y_preds])
                all_classes = classes_true.union(classes_pred)
                
                for c in sorted(list(all_classes)):
                    c = int(c)
                    if c not in self.classes:
                        self.classes.append(c)
                        self.tp[c] = 0
                        self.tn[c] = 0
                        self.fp[c] = 0
                        self.fn[c] = 0
                        self.total_pos[c] = 0
                    (batch_tp, batch_tn, batch_fp,
                         batch_fn, batch_total_pos) = _compute_tp_tn_fp_fn_total_pos(
                            Y_true, Y_preds, self.using_np, c
                        )
                    self.tp[c] += batch_tp
                    self.tn[c] += batch_tn
                    self.fp[c] += batch_fp
                    self.fn[c] += batch_fn
                    self.total_pos[c] += batch_total_pos
            
    def compute(self):
        '''
        Computes metrics from the running batches.
        
        Returns:
            List containing pairs with the computed metrics. Each pair has
            the name of the metric as its first entry (as determined by the
            metrics argument), and the value of the metric as its second
            entry. The order of the metrics is the same as given in the
            metrics argument.
        '''

        computing_f1 = False
        if 'f1' in self.metrics:
            if self.num_classes == 1:
                pre = [_compute_pre(self.tp, self.tn, self.fp, self.fn, self.total_pos)]
                rec = [_compute_rec(self.tp, self.tn, self.fp, self.fn, self.total_pos)]
            else:
                pre = [
                    _compute_pre(self.tp[c], self.tn[c], self.fp[c],
                                 self.fn[c], self.total_pos[c])
                    for c in self.classes
                ]
                rec = [
                    _compute_rec(self.tp[c], self.tn[c], self.fp[c],
                                 self.fn[c], self.total_pos[c])
                    for c in self.classes
                ]
            computing_f1 = True

        results = []
        if self.num_classes > 1:
            results.append(('classes', self.classes))
        for metric in self.metrics:
            if metric == 'acc':
                results.append((
                    metric, _compute_acc_stats(self.num_correct, self.total)))
            if metric == 'pre':
                if computing_f1:
                    results.append((metric, pre))
                else:
                    if self.num_classes == 1:
                        results.append((
                            metric, _compute_pre(self.tp, self.tn, self.fp, self.fn, self.total_pos)))
                    else:
                        results.append((
                            metric, [
                                _compute_pre(self.tp[c], self.tn[c], self.fp[c],
                                             self.fn[c], self.total_pos[c])
                                for c in self.classes
                            ]))
            if metric == 'rec':
                if computing_f1:
                    results.append((metric, rec))
                else:
                    if self.num_classes == 1:
                        results.append((
                            metric, _compute_rec(self.tp, self.tn, self.fp, self.fn, self.total_pos)))
                    else:
                        results.append((
                            metric, [
                                _compute_rec(self.tp[c], self.tn[c], self.fp[c],
                                             self.fn[c], self.total_pos[c])
                                for c in self.classes
                            ]))
            if metric == 'f1':
                if self.num_classes == 1:
                    results.append((metric, _compute_f1(pre, rec)))
                else:
                    results.append((metric, [
                        _compute_f1(p, r)
                        for p, r in zip(pre, rec)
                    ]))
            if metric in ['tp', 'tn', 'fp', 'fn']:
                results.append((
                    metric, getattr(self, metric) if self.num_classes == 1 else [
                        getattr(self, metric)[cls]
                        for cls in self.classes
                    ]
                ))

        return results

def compute_metrics(
    Y_true: NpOrTensor,
    Y_preds: NpOrTensor,
    metrics: Optional[Sequence[str]] = ['acc', 'pre', 'rec', 'f1'],
    pos_class: Optional[int] = 1,
    abstain_val: Optional[int] = None,
    num_classes: Optional[int] = 1,
) -> List[Tuple[str, float]]:
    '''
    Compute validation metrics for Y_preds based on ground-truth labels
    Y_true.
    Can take in either Numpy arrays or PyTorch tensors (but Y_true and
    Y_preds have to use the same one).
    If PyTorch tensors are used, it is possible to do the computation
    on the GPU (but results will be moved to DRAM to be returned).
    Returns a list with the computed metrics.
    
    Currently only top-1 accuracy, precision, recall, and F1 supported.
    For precision, recall, and F1, only binary cases supported, with
    1 as the binary class.
    
    Args:
        Y_true (Numpy array/PyTorch tensor): Ground-truth labels.
            Can either be a Numpy array or a PyTorch tensor.
        Y_preds (Numpy array/PyTorch tensor): Predictions.
            Can be a Numpy array or a PyTorch tensor, but needs to
            have the same type as Y_true.
            If a PyTorch tensor, needs to be on the same device (GPU
            or DRAM) as Y_true.
        metrics (Sequence[str]): List of metrics to compute. Currently,
            'acc', 'pre', 'rec', and 'f1' are supported, and all of
            these are computed by default.
            acc is top-1 accuracy.
            pre, rec, and f1 are precision, recall, and f1 for a binary
                classifier. These metrics assume that 1 is the label of
                the positive class.
    
    Returns:
        List containing pairs with the computed metrics. Each pair has
        the name of the metric as its first entry (as determined by the
        metrics argument), and the value of the metric as its second
        entry. The order of the metrics is the same as given in the
        metrics argument.
    '''
    stream_metrics = StreamMetrics(metrics, pos_class, abstain_val, num_classes)
    
    stream_metrics.update(Y_true, Y_preds)
    
    return stream_metrics.compute()

def _is_using_np(
    Y_true: NpOrTensor,
    Y_preds: NpOrTensor,
) -> bool:
    '''Figure out whether we are using Numpy arrays, or throw an error if something
    is inconsistent.'''
    if _is_numpy(Y_true):
        using_np = True
    elif _is_torch(Y_true):
        using_np = False
    else:
        raise NotImplementedError('Y_true must be a Numpy array or a PyTorch tensor')
        
    if using_np and not _is_numpy(Y_preds):
        raise NotImplementedError('Y_true is Numpy, but Y_preds is not')
    elif not using_np and not _is_torch(Y_preds):
        raise NotImplementedError('Y_true is PyTorch, but Y_preds is not')
    
    return using_np

def _is_numpy(arr: NpOrTensor):
    '''Check if arr is a Numpy array.'''
    return isinstance(arr, type(np_arr))

def _is_torch(arr: NpOrTensor):
    '''Check if arr is a PyTorch tensor.'''
    return isinstance(arr, type(torch_arr))

def _compute_corrects(labels: NpOrTensor, preds: NpOrTensor, using_np: bool,
                     abstain_val: Optional[int] = None) -> int:
    if abstain_val is None:
        num_correct = (labels == preds).sum()
    else:
        num_correct = (labels == preds)[preds != abstain_val].sum()
    if not using_np:
        num_correct = num_correct.item()
        
    return num_correct

# def _compute_acc(labels: NpOrTensor, preds: NpOrTensor, using_np: bool) -> float:
#     return _compute_corrects(labels, preds, using_np) / labels.shape[0]

def _compute_acc_stats(corrects: int, total: int) -> float:
    return corrects / total

def _compute_tp_tn_fp_fn_total_pos(
    labels: np.ndarray, preds: np.ndarray, using_np: bool, pos_class: int,
    abstain_val: Optional[int] = None
) -> Tuple[int, int, int, int]:
    correct = labels == preds
    incorrect = labels != preds
    pos_labels = labels == pos_class
    neg_labels = labels != pos_class
    pos_preds = preds == pos_class
    neg_preds = preds != pos_class
    
    if abstain_val is None:
        tp = (correct & pos_labels).sum()
        tn = (correct & neg_labels).sum()
        fp = (incorrect & pos_preds).sum()
        fn = (incorrect & pos_labels).sum()
    else:
        voting_indices = preds != abstain_val
        tp = (correct & pos_labels)[voting_indices].sum()
        tn = (correct & neg_labels)[voting_indices].sum()
        fp = (incorrect & pos_preds)[voting_indices].sum()
        fn = (incorrect & pos_labels)[voting_indices].sum()
    total_pos = pos_labels.sum()
    
    if not using_np:
        tp = tp.item()
        tn = tn.item()
        fp = fp.item()
        fn = fn.item()
        total_pos = total_pos.item()
    
    return tp, tn, fp, fn, total_pos

def _compute_pre(tp: int, tn: int, fp: int, fn: int, total_pos: int) -> float:
    return tp / (tp + fp) if tp + fp > 0 else 0.

def _compute_rec(tp: int, tn: int, fp: int, fn: int, total_pos: int) -> float:
    return tp / total_pos if total_pos > 0 else 0.

def _compute_f1(pre: float, rec: float) -> float:
    return 2 * pre * rec / (pre + rec) if pre + rec > 0 else 0.