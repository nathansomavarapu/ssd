import argparse
from eval.voc_eval import voc_eval
from runner import ModelRunner

import cv2
import utils
import glob
import os
import shutil
import tqdm

from PIL import Image

import numpy as np


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def evaluate_voc(image_path='../data/VOC2007/test/JPEGImages/', anno_path='../data/VOC2007/test/Annotations/', image_set_path='../data/VOC2007/test/ImageSets/Main/', class_path='../data/VOC2007/train/classes.txt', clean=True):

    imgs = sorted(glob.glob(os.path.join(image_path, '*.jpg')))

    anno_path = os.path.join(anno_path, '{}.xml')
    image_set_path = os.path.join(image_set_path, '{}_test.txt')

    int_to_cl = []

    with open(class_path, 'r') as f:
        for line in f:
            int_to_cl.append(line.strip())
    
    base = os.path.join('eval', 'voc_dets')

    if not os.path.exists(base):
        os.makedirs(base)

    print('Computing Image Detections...')

    for img_f in tqdm.tqdm(imgs):
        img = Image.open(img_f)
        w, h = img.size

        preds = runner.run_inference(img, convert=False)

        img_num = img_f.split('/')[-1].replace('.jpg', '')
        

        for pred in preds:
            outfile = os.path.join(base, int_to_cl[int(pred[0]) - 1] + '.txt')

            line = '%s %.3f %.2f %.2f %.2f %.2f' % (img_num, pred[1], pred[2].item(
            ) * w, pred[3].item() * h, pred[4].item() * w, pred[5].item() * h)

            with open(outfile, 'a+') as f:
                f.write(line + '\n')

    print('Computng APs...')
    cl_to_ap = {}
    for cl in tqdm.tqdm(int_to_cl):
        curr_det_path = os.path.join(base, '{}.txt')
        
        if os.path.exists(curr_det_path.format(cl)):

            recall, precision, ap = voc_eval(curr_det_path, anno_path, image_set_path.format(cl), cl, base, use_07_metric=True)
            cl_to_ap[cl] = ap
    

    for k,v in cl_to_ap.items():
        print('AP for ' + k + ': ' + str(v))
    
    print('')
    print('')
    print('MAP: ' + str(np.mean(np.array(list(cl_to_ap.values())))))

    if clean:
        shutil.rmtree(base)


def run_webcam():
    pass


def evaluate_coco():
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Evaluate a model, or run it live.')
    parser.add_argument('-wp', '--weights_path', type=str,
                        required=True, help='Path to weights for the model')
    parser.add_argument('-bn', '--batchnorm', type=str2bool,
                        default=False, help='Use batchnorm')
    parser.add_argument('-b', '--base', type=str,
                        default='vgg', help='Model feature extractor')
    parser.add_argument('-nc', '--num_classes', type=int,
                        required=True, help='Number of classes')
    parser.add_argument('-f', '--function', type=str,
                        help='video, eval_voc or eval_coco')
    args = parser.parse_args()

    runner = ModelRunner(args.weights_path, args.base,
                         args.batchnorm, num_cats=args.num_classes)

    if args.function.lower() == 'video':
        cam = cv2.VideoCapture(0)
        while(cam.isOpened()):
            ret, frame = cam.read()

            if ret == True:

                preds = runner.run_inference(frame, convert=True)

                frame = utils.draw_bbx(frame, preds, [0, 0, 255], pil=False)

                cv2.imwrite('face3.jpg', frame)

                cv2.imshow('Demo', frame)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
    elif args.function.lower() == 'eval_voc':

        evaluate_voc()

    elif args.function.lower() == 'eval_coco':

        annFile = '../data/annotations2014/instances_val2014.json'
        cocoGt = COCO(annFile)

        samples_path = '../data/train2014'
        imgs = glob.glob(os.path.join(samples_path, '*.jpg'))

        det_arr = []
        img_ids = []
        time_per_inf = 0
        for i, image_path in enumerate(tqdm.tqdm(imgs)):
            image = Image.open(image_path)

            img_id = image_path.split('/')[-1].split('_')[-1].lstrip('0')
            img_id = int(img_id.replace('.jpg', ''))

            s = time.time()
            preds = runner.run_inference(img, convert=False)
            e = time.time()

            if i > 0:
                time_per_inf += abs(e-s)

            for pred in preds:
                xmin, ymin, xmax, ymax = (int(pred[2].item(
                ) * w), int(pred[3].item() * h), int(pred[4].item() * w), int(pred[5].item() * h))

                curr_det = []
                curr_det.append(img_id)
                curr_det.extend([xmin, ymin, ymax - ymin, xmax - xmin])
                curr_det.append(pred[1])
                curr_det.append(pred[0])
                det_arr.append(curr_det)

            img_ids.append(img_id)

        print('Average forward infrence time per image for running over ' +
              str(len(imgs)) + ' images: ' + str(time_per_inf/(len(imgs)-1)) + 's')

        cocoDt = cocoGt.loadRes(np.array(det_arr))
        imgIds = np.array(img_ids)
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
