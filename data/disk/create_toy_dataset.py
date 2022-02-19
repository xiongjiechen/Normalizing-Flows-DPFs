import argparse
import os
import cv2
import logging
import sys
import numpy as np
# from contexts import recordio as tfr

class ToyExample():
    def __init__(self, param):
        self.im_size = param.width
        self.out_dir = param.out_dir
        self.name = param.name

        self.num_examples = param.num_examples
        self.sequence_length = param.sequence_length
        self.file_size = min(self.num_examples, param.file_size)

        self.spring_force = 0.1 # origin:0.05, #0.1  # 0.1 for one object; 0.05 for five objects
        self.drag_force = 0.0075

        self.cols = [(0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255),
                     (255, 255, 0), (255, 255, 255)]  # no black, no red.

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        # setup logging
        self.log = logging.getLogger(param.name)
        self.log.setLevel(logging.DEBUG)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s: [%(name)s] ' +
                                      '[%(levelname)s] %(message)s')
        # create console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.log.addHandler(ch)

        # create file handler which logs warnings errors and criticals
        if os.path.exists(os.path.join(self.out_dir,
                                       self.name + '_error.log')):
            os.remove(os.path.join(self.out_dir,
                                   self.name + '_error.log'))
        fh = logging.FileHandler(os.path.join(self.out_dir,
                                              self.name + '_error.log'))
        fh.setLevel(logging.WARNING)
        fh.setFormatter(formatter)
        self.log.addHandler(fh)

    def create_dataset(self, num_distractors, pos_noise):

        # # tfread file setting
        # self.record_writer_train = \
        #     tfr.RecordioWriter(self.out_dir, self.file_size,
        #                        self.name + '_train_')
        # self.record_meta_train = tfr.RecordMeta(self.name + '_train_')
        # self.record_writer_val = \
        #     tfr.RecordioWriter(self.out_dir, self.file_size,
        #                        self.name + '_val_')
        # self.record_meta_val = tfr.RecordMeta(self.name + '_val_')
        # self.record_writer_test = \
        #     tfr.RecordioWriter(self.out_dir, self.file_size,
        #                        self.name + '_test_')
        # self.record_meta_test = tfr.RecordMeta(self.name + '_test_')

        self.keys = ['start_image', 'start_state', 'image', 'state', 'q',
                     'visible']
        train_data = {key: [] for key in self.keys}

        self.log.info('Starting to generate dataset ' + self.name)

        train_count = 0
        val_count = 0
        test_count = 0

        ct=0
        index=0
        while train_count < self.num_examples:
            values = self._get_data(num_distractors, pos_noise)
            ct+=1
            for key in self.keys:
                train_data[key] += [values[key]]
            if len(train_data['image']) >= self.file_size:
                train_size, val_size, test_size = self._save(train_data, index)
                train_count += train_size
                val_count += val_size
                test_count += test_size
                index+=1
                train_data = {key: [] for key in self.keys}

            if ct % 250 == 0:
                self.log.info('Done ' + str(ct) +
                              ' of ' + str(int(self.num_examples/0.8)))
        if len(train_data['image']) > 0:
            train_size, val_size, test_size = self._save(train_data)
            train_count += train_size
            val_count += val_size
            test_count += test_size

        # save the meta information
        count = train_count + val_count + test_count
        fi = open(os.path.join(self.out_dir, 'info_' + self.name + '.txt'),
                  'w')
        fi.write('Num data points: ' + str(count) + '\n')
        fi.write('Num train: ' + str(train_count) + '\n')
        fi.write('Num val: ' + str(val_count) + '\n')
        fi.write('Num test: ' + str(test_count) + '\n')

        fi.close()

        self.log.info('Done')

        # self.record_writer_train.close()
        # self.record_writer_test.close()
        # self.record_writer_val.close()

        return

    def _get_data(self, num_distractors, pos_noise):
        states = []
        images = []
        qs = []
        rs = []

        # the state consists of the red disc's position and velocity
        # draw a random position
        pos = np.random.uniform(-self.im_size // 2, self.im_size // 2, size=(2))
        # draw a random velocity
        vel = np.random.normal(loc=0, scale=1, size=(2)) * 3
        initial_state = np.array([pos[0], pos[1], vel[0], vel[1]])

        distractors = []
        for dist in range(num_distractors):
            # also draw a random starting positions for distractors
            pos = np.random.uniform(-self.im_size // 2, self.im_size // 2,
                                    size=(2))
            # draw a random velocity
            vel = np.random.normal(loc=0, scale=1, size=(2)) * 3
            # and a random radius
            rad = np.random.choice(np.arange(3, 10))
            # draw a random color
            col = np.random.choice(len(self.cols))
            distractors += [(rad, np.array([pos[0], pos[1], vel[0], vel[1]]),
                             col)]

        # generate the initial image
        initial_im, initial_vis = self._observation_model(initial_state,
                                                          distractors)
        last_state = initial_state
        for step in range(self.sequence_length):
            # get the next state
            state, q = self._process_model(last_state, pos_noise)
            # also move the distractors
            new_distractors = []
            for d in distractors:
                d_new, _ = self._process_model(d[1], pos_noise)
                new_distractors += [(d[0], d_new, d[2])]
            # get the new image
            im, vis = self._observation_model(state, new_distractors)

            states += [state]
            images += [im]
            qs += [q]
            rs += [vis]
            distractors = new_distractors
            last_state = state

        return {'start_image': initial_im, 'start_state': initial_state,
                'image': np.array(images), 'state': np.array(states),
                'q': np.array(qs), 'visible': np.array(rs)}

    def _observation_model(self, state, distractors):
        im = np.zeros((self.im_size, self.im_size, 3))

        # draw the red disc
        cv2.circle(im, (int(state[0] + self.im_size // 2),
                        int(state[1] + self.im_size // 2)),
                   radius=7, color=[255, 0, 0], thickness=-1)

        # draw the other distractors
        for d in distractors:
            cv2.circle(im, (int(d[1][0] + self.im_size // 2),
                            int(d[1][1] + self.im_size // 2)),
                       radius=d[0], color=self.cols[d[2]], thickness=-1)  # thickness=-1 means filled circle

        # get the number of pixels visible from the red disc
        mask = np.logical_and(im[:, :, 0] == 255,
                              np.logical_and(im[:, :, 1] == 0,
                                             im[:, :, 2] == 0))

        vis = np.sum(mask)
        im = im.astype(np.float32) / 255.
        # vis is used to judge whether the disk is out of the scene
        return im, vis

    def _process_model(self, state, pos_noise):
        new_state = np.copy(state)
        pull_force = - self.spring_force * state[:2]
        drag_force = - self.drag_force * state[2:] ** 2 * np.sign(state[2:])
        new_state[0] += state[2]
        new_state[1] += state[3]
        new_state[2] += pull_force[0] + drag_force[0]
        new_state[3] += pull_force[1] + drag_force[1]

        position_noise = np.random.normal(loc=0, scale=pos_noise, size=(2))

        velocity_noise = np.random.normal(loc=0, scale=4.,
                                          size=(2))
        q = 2. # the std of velocity noise
        new_state[:2] += position_noise
        new_state[2:] += velocity_noise

        q = np.array([pos_noise, pos_noise, q, q])

        return new_state, q

    def _save(self, data, index):
        length = len(data['image'])
        # convert lists to numpy arrays
        for key in self.keys:
            if type(data[key]) == np.ndarray and data[key].dtype == np.float64:
                data[key] = np.array(data[key]).astype(np.float32)

        # shuffle the arrays together
        permutation = np.random.permutation(length)
        for key in self.keys:
            vals = np.copy(data[key])
            data[key] = vals[permutation]

        # 80% for train; 10% for val; 10% for test
        train_size = int(np.floor(length * 8. / 10.))
        val_size = int(np.floor(length * 1. / 10.))
        test_size = length - train_size - val_size

        if train_size > 0:
            train_data = {}
            for key in self.keys:
                train_data[key] = np.copy(data[key][:train_size])
            np.savez(os.path.join(self.out_dir, self.name+str(index)+"_train.npz"), train_data=train_data)

        if val_size > 0:
            val_data = {}
            for key in self.keys:
                val_data[key] = \
                    np.copy(data[key][train_size:train_size+val_size])
            np.savez(os.path.join(self.out_dir, self.name+str(index)+"_val.npz"), val_data=val_data)

        if test_size > 0:
            test_data = {}
            for key in self.keys:
                test_data[key] = np.copy(data[key][train_size+val_size:])
            np.savez(os.path.join(self.out_dir, self.name+str(index)+"_test.npz"), test_data=test_data)

        return train_size, val_size, test_size

        # if train_size > 0:
        #     train_data = {}
        #     for key in self.keys:
        #         train_data[key] = np.copy(data[key][:train_size])
        #     rw = self.record_writer_train
        #     rm = self.record_meta_train
        #     tfr.write_tfr(train_data, rw, rm, self.out_dir)
        #
        # if val_size > 0:
        #     val_data = {}
        #     for key in self.keys:
        #         val_data[key] = \
        #             np.copy(data[key][train_size:train_size+val_size])
        #     rw = self.record_writer_val
        #     rm = self.record_meta_val
        #     tfr.write_tfr(val_data, rw, rm, self.out_dir)
        #
        # if test_size > 0:
        #     test_data = {}
        #     for key in self.keys:
        #         test_data[key] = np.copy(data[key][train_size+val_size:])
        #     rw = self.record_writer_test
        #     rm = self.record_meta_test
        #     tfr.write_tfr(test_data, rw, rm, self.out_dir)
        # return train_size, val_size, test_size

def main():
    parser = argparse.ArgumentParser('toy datset')
    parser.add_argument('--name', dest='name', type=str, default='toy')
    parser.add_argument('--out-dir', dest='out_dir', type=str, default='./TwentyfiveDistractors',
                        help='where to store results')
    parser.add_argument('--sequence-length', dest='sequence_length', type=int,
                        default=50, help='length of the generated sequences')
    parser.add_argument('--width', dest='width', type=int, default=128,
                        help='width (= height) of the generated observations')
    parser.add_argument('--num-examples', dest='num_examples', type=int,
                        default=1000,
                        help='how many training examples should be generated')
    # 2400 for train; 300 for val; 300 for test, num_examples= 3000
    # 1200 for train; 150 for val; 150 for test, num_examples= 1000
    parser.add_argument('--file-size', dest='file_size', type=int,
                        default=500,
                        help='how many examples per file should be saved in ' +
                             'one record')

    parser.add_argument('--pos-noise', dest='pos_noise', type=float,
                        default=0.1,
                        help='sigma for the positional process noise')
    parser.add_argument('--num-distractors', dest='num_distractors', type=int,
                        default=25, help='number of distractor disc') # orig: 5

    args = parser.parse_args()

    name = args.name + '_pn=' + str(args.pos_noise) \
           + '_d=' + str(args.num_distractors)

    name += '_const'

    args.name = name

    if not os.path.exists(os.path.join(args.out_dir,
                                       'info_' + args.name + '.txt')):
        c = ToyExample(args)
        c.create_dataset(args.num_distractors, args.pos_noise)
    else:
        print('name already exists')


if __name__ == "__main__":
    main()