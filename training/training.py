# coding='utf-8'
import os
import sys
import argparse
import numpy as np
import time
import datetime

sys.path.insert(0, "/home/liubofang/bob_opensource/incubator-mxnet-v1.1.0/python")
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..', 'common'))
from batch_reader import BatchReader
import models

def train(prefix, **arg_dict):
    num_labels = arg_dict['landmark_type'] * 2
    img_size = arg_dict['img_size']
    train_angle = arg_dict['train_angle']
    gpu_num = len(arg_dict["gpu_device"].split(','))
    batch_size = arg_dict['batch_size'] * gpu_num
    arg_dict['batch_size'] = batch_size
    print ("real batch_size = %d for gpu_num = %d" % (batch_size, gpu_num))
    # batch generator
    _batch_reader = BatchReader(**arg_dict)
    _batch_generator = _batch_reader.batch_generator()
    # net
    ctx = [mx.gpu(i) for i in range(gpu_num)]
    net =  models.init(num_label=num_labels, **arg_dict)
    if arg_dict["restore_ckpt"]:
        print "resotre checkpoint from %s" % (arg_dict["restore_ckpt"])
        net.load_params(arg_dict['restore_ckpt'], ctx=ctx)
    else:
        net.initialize(init=mx.init.Xavier(), ctx=ctx)
    print net
    # loss
    losses_func = []
    if train_angle:
        losses_func.append(gluon.loss.L2Loss(weight=0.5))  # landmark
        losses_func.append(gluon.loss.L2Loss(weight=0.5))  # angle
    else:
        losses_func.append(gluon.loss.L2Loss())  # landmark
    # trainer
    trainer = gluon.Trainer(net.collect_params(), "adam",
                            {"learning_rate": arg_dict['learning_rate']})
    # start loop
    print ("Start to training...")
    start_time = time.time()
    step = 0
    display = 10
    loss_list = []
    while not _batch_reader.should_stop():
        batch = _batch_generator.next()
        image = nd.array(batch[0])
        image = nd.transpose(image.astype('float32'), (0,3,1,2)) / 127.5 - 1.0
        image_list = gluon.utils.split_and_load(image, ctx)
        landmark = nd.array(batch[1])
        landmark_list = gluon.utils.split_and_load(landmark, ctx)
        if train_angle:
            angle = nd.array(batch[2])
            angle_list = gluon.utils.split_and_load(angle, ctx)
        with autograd.record():
            losses = []
            if train_angle:
                for _i, _l, _a in zip(image_list, landmark_list, angle_list):
                    predicts = net(_i)
                    landmark_loss = losses_func[0](predicts[0], _l)
                    angle_loss = losses_func[1](predicts[1], _a)
                    losses.append(landmark_loss + angle_loss)
            else:
                for _i, _l in zip(image_list, landmark_list):
                    predicts = net(_i)
                    landmark_loss = losses_func[0](predicts, _l)
                    losses.append(landmark_loss)
        for loss in losses:
            loss.backward()
        trainer.step(batch_size)
        loss_list.append(np.mean([nd.mean(l).asscalar() for l in losses]))
        nd.waitall()
        if step % display == 0:
            end_time = time.time()
            cost_time, start_time = end_time - start_time, end_time
            sample_per_sec = int(display * batch_size / cost_time)
            sec_per_step = cost_time / float(display)
            loss_display = "[landmark: %.5f]" % (np.mean(loss_list))
            print ('[%s] epochs: %d, step: %d, lr: %.5f, loss: %s,'\
                   'sample/s: %d, sec/step: %.3f' % (
                   datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), 
                   _batch_reader.get_epoch(), step, trainer.learning_rate, loss_display,
                   sample_per_sec, sec_per_step))
            loss_list = []
        if step % 1024 == 0:
            # change lr
            trainer.set_learning_rate(trainer.learning_rate * 0.95)
            # save checkpoint
            checkpoint_path = os.path.join(prefix, 'model.params')
            net.save_params(checkpoint_path)
            print ("save checkpoint to %s" % checkpoint_path)
        step += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_paths', type=str, nargs='+',
                        default='/world/data-c9/liubofang/dataset_original/CelebA/full_path_zf_bbox_pts.txt')
    parser.add_argument('--working_root', type=str, default='/world/data-c9/liubofang/training/landmarks/celeba')
    parser.add_argument('--batch_size', type=int, default=512, help="Batch size for training")
    parser.add_argument('--landmark_type', type=int, default=5, help="The number of points. 5, 72, 83")
    parser.add_argument('--max_epoch', type=int, default=1000, help="Training will be stoped in this case.")
    parser.add_argument('--img_size', type=int, default=128, help="The size of input for model")
    parser.add_argument('--max_angle', type=int, default=10, help="Use for image augmentation")
    parser.add_argument('--process_num', type=int, default=20, help="The number of process to preprocess image.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="lr")
    parser.add_argument('--model', type=str, default='fanet8ss_inference', help="Model name. Check models.py")
    parser.add_argument('--restore_ckpt', type=str, help="Resume training from special ckpt.")
    parser.add_argument('--try', type=int, default=0, help="Saving path index")
    parser.add_argument('--gpu_device', type=str, default='7', help="GPU index")
    parser.add_argument('--img_format', type=str, default='RGB', help="The color format for training.")
    parser.add_argument('--train_angle', type=bool, default=False, help="angle loss.")
    parser.add_argument('--buffer2memory', type=bool, default=False, 
                        help="Read all image to memory to speed up training. Make sure enough memory in your device.")
    arg_dict = vars(parser.parse_args())
    prefix = '%s/%d/%s/size%d_angle%d_try%d' % (
        arg_dict['working_root'], arg_dict['landmark_type'], arg_dict['model'], 
        arg_dict['img_size'], arg_dict['max_angle'], arg_dict['try'])
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    # set up environment
    os.environ['CUDA_VISIBLE_DEVICES']=arg_dict['gpu_device']

    train(prefix, **arg_dict)

if __name__ == "__main__":
    main()

