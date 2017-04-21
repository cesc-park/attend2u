from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import colorlog
import pprint

from utils.pycocoevalcap.eval import COCOEvalCap
from utils.pycocotools.coco import COCO
from utils.utils import nostdout

pp = pprint.PrettyPrinter().pprint


class Evaluator(object):
  def __init__(self):
    pass

  def evaluation(self, predicts, answers, method="coco"):
    """Wrapper method for evaluation
    Args:
        predicts: list of tokens list
        answers: list of tokens list
        method: evaluation method. ("coco")
    Returns:
        Dictionary with metric name in key metric result in value
    """
    colorlog.info("Run evaluation...")
    if method == "coco":
        eval_result = self._coco_evaluation(predicts, answers)
    else:
        raise NotImplementedError

    pp(eval_result)
    return eval_result

  def _coco_evaluation(self, predicts, answers):
    coco_res = []
    ann = {
        'images': [], 'info': '', 'type': 'captions',
        'annotations': [], 'licenses': ''
    }

    for i, (predict, answer) in enumerate(zip(predicts, answers)):
      predict_cap = ' '.join(predict)
      answer_cap = ' '.join(answer).replace('_UNK', '_UNKNOWN')

      ann['images'].append({'id': i})
      ann['annotations'].append(
          {'caption': answer_cap, 'id': i, 'image_id': i}
      )
      coco_res.append(
          {'caption': predict_cap, 'id': i, 'image_id': i}
      )

    with nostdout():
      coco = COCO(ann)
      coco_res = coco.loadRes(coco_res)
      coco_eval = COCOEvalCap(coco, coco_res)
      coco_eval.evaluate()

    return coco_eval.eval
