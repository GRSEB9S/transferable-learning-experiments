from lib.alexnet import AlexNet
from lib.data import Data
from lib.perf import Perf
import lib.utils as utils
import lib.log as log
import numpy as np
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

# to configure
IDENTIFIER = 'eye_net'

with AlexNet() as alexnet:
    # Load data
    log.log("[Loading Data...]")
    all_data = Data()
    all_data.load_images('data/Flickr_2800/eye_2800', POSITIVE_LABEL)
    all_data.load_images('data/Flickr_2800/notall', NEGATIVE_LABEL)
    train_data, test_data = all_data.split_train_test(train_split=0.90)

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
    overall_perf.graph()

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
