from lib.alexnet import AlexNet
from lib.data import Data
from lib.perf import Perf
import lib.utils as utils
import lib.log as log
import numpy as np
import argparse
import os

# training params
training_iters = 10000
batch_size = 50
display_step = 250

# constants
POSITIVE_LABEL = np.array([1, 0])
NEGATIVE_LABEL = np.array([0, 1])
PERF_DIR = 'results/perf'
CHECKPOINT_DIR = 'results/checkpoint'
TRANSPLANT_DIR = 'results/transfer'

# globals
perfs = []

# actual run
if __name__ == "__main__":
    # parse args out
    parser = argparse.ArgumentParser()
    parser.add_argument("identifier", type=str, help="A uniquely identifying string for this file with format [init_weights_class]_[training_class]")
    parser.add_argument("pos_class", type=str, help="Which folder in data/Flickr_2800/ to load for training")
    parser.add_argument('-t', '--transplant', type=str, help='The identifier of the transplant to start from')
    parser.add_argument('-l', '--lesion_indicator', type=str, default='')
    args = parser.parse_args()

    # load user configured constants
    if args.lesion_indicator is not '':
        IDENTIFIER = args.identifier + '_t' + args.lesion_indicator + utils.datestr()
    else:
        IDENTIFIER = args.identifier + utils.datestr()

    POSITIVE_IMAGE_DIR = os.path.join('data/Flickr_2800', args.pos_class)

    # let the user know what we're up to
    log.log("Training " + IDENTIFIER + " FROM " + POSITIVE_IMAGE_DIR)
    if args.transplant is not None:
        log.log("(Starting from " + args.transplant + "_t" + args.lesion_indicator + ")")
    else:
        log.log("(Starting from <blank net, random weights>)")

    # train it all
    with AlexNet() as alexnet:
        # Load data
        log.log("[Loading Data...]")
        all_data = Data()
        all_data.load_images(POSITIVE_IMAGE_DIR, POSITIVE_LABEL)
        all_data.load_images('data/Flickr_2800/notall', NEGATIVE_LABEL)
        train_data, test_data = all_data.split_train_test(train_split=0.90)

        # Initialize net
        if args.transplant is not None:
            load_transplant_dir = os.path.join(TRANSPLANT_DIR, args.transplant + '.ckpt')
            alexnet.load_transplant(load_transplant_dir)

        if args.lesion_indicator is not '':
            layers = [i for i, e in enumerate(LESION_INDICATORS) if e == '0']
            alex_net.lesion_layers(layers)

        # Train
        log.log("[Training...]")
        i = 0
        while i < training_iters:
            i += batch_size

            # train the net
            batch = train_data.get_batch(batch_size)
            alexnet.train(batch)

            # print perf
            if i % display_step == 0:
                # print what we're up to
                log.log_ephemeral('(Evaluating)     ')

                # compute perf
                curr_perf = alexnet.measure_perf(test_data.get_all())
                curr_perf['step'] = i

                # save it
                perfs.append(curr_perf)

                # log it
                log.log(curr_perf)

            else:
                # print how long until next perf
                done = float(i % display_step) / display_step
                log.log_ephemeral('({0}% Trained)'.format(done * 100))

        # construct perf
        overall_perf = Perf(perfs)

        # save results and data
        utils.ensure_path(PERF_DIR)
        utils.ensure_path(CHECKPOINT_DIR)
        utils.ensure_path(TRANSPLANT_DIR)

        perf_file = os.path.join(PERF_DIR, IDENTIFIER + '.csv')
        checkpoint_file = os.path.join(CHECKPOINT_DIR, IDENTIFIER + '.ckpt')
        transplant_file = os.path.join(TRANSPLANT_DIR, IDENTIFIER + '.ckpt')

        overall_perf.save(perf_file)
        alexnet.save_transplant(transplant_file)
        alexnet.save_full_checkpoint(checkpoint_file)
