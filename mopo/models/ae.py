import numpy as np
import tensorflow as tf
import pdb
import itertools
from collections import OrderedDict

from mopo.models.utils import get_required_argument
from mopo.utils.logging import Progress, Silent

import wandb

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class AE:
    """ The energy model with sliced score matching. """

    def __init__(self, lr,
                 obs_dim, act_dim, rew_dim, hidden_dim,
                 early_stop_patience, name="SSM"):
        self.lr = lr

        self._early_stop_patience = early_stop_patience
        self._state = {}  # used for storing weight
        self._snapshot = (None, 1e10)  # store the best (epoch, val_loss) pair
        self._epochs_since_update = 0
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.rew_dim = rew_dim
        self.hidden_dim = hidden_dim

        self.noise_type = 'radermacher'

        self.sess_ssm = tf.Session(config=tf.ConfigProto())

    def get_data(self):
        # create placeholder for the graph
        self.input = tf.placeholder(tf.float32,
                                    shape=[None, self.obs_dim * 2 + self.act_dim],  # [batch_size, data_size]
                                    name="inputs_data")

        self.label = tf.placeholder(tf.float32,
                                    shape=[None, self.obs_dim * 2 + self.act_dim],
                                    name='output_data')

    def inference(self):
        # Build model here, predict the energy
        fc1 = tf.layers.dense(self.input,
                              self.hidden_dim,
                              activation=tf.nn.softplus,
                              name='fc1')  # Output tensor the same shape as inputs except the last dimension is of size units.
        fc2 = tf.layers.dense(fc1,
                              self.hidden_dim//4,
                              activation=tf.nn.softplus,
                              name='fc2')
        # fc3 = tf.layers.dense(fc2,
        #                       self.hidden_dim//4,
        #                       activation=tf.nn.softplus,
        #                       name='fc3')
        # fc4 = tf.layers.dense(fc3,
        #                       self.hidden_dim//2,
        #                       activation=tf.nn.softplus,
        #                       name='fc4')
        fc5 = tf.layers.dense(fc2,
                              self.hidden_dim,
                              activation=tf.nn.softplus,
                              name='fc5')
        self.pred = tf.layers.dense(fc5,
                                      self.obs_dim * 2 + self.act_dim,
                                      name='prediction')  # returned shape should be  [ batch_size, obs_dim * 2 + act_dim]

    def loss(self):
        with tf.variable_scope('loss'):
            # self.loss = tf.reduce_mean(tf.squared_difference(self.pred, self.label), name='l2_loss')
            loss = tf.nn.l2_loss(self.pred-self.label, name='l2_loss')
            self.loss = tf.reduce_mean(loss)

    def optimize(self):
        # setup the optimizer, using Adam
        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def eval(self, val_in, val_label):
        with tf.variable_scope('eval'):
            eval_loss = self.sess_ssm.run(
                self.loss,
                feed_dict={
                    self.input: val_in, self.label: val_label
                }
            )

            # l1 = self.sess_ssm.run(
            #     self.loss1,
            #     feed_dict={self.input: val_in}
            # )
            # l2 = self.sess_ssm.run(
            #     self.loss2,
            #     feed_dict={self.input: val_in}
            # )
        # print("during eval, the loss 1 is {}, the loss 2 is {}".format(l1, l2))
        return eval_loss

    def summary(self):
        # TODO
        pass

    def _predict(self):
        with tf.variable_scope('predict'):
            self.reconstruct_error = tf.reduce_sum(tf.squared_difference(self.pred, self.label), axis=-1, keep_dims=True, name='reconstruct_error')

    def predict(self, inputs):
        # inputs = np.tile(inputs[None], [self.num_nets, 1, 1]) # to align the shape
        # return self.sess_ssm.run(self.energy, feed_dict={self.input: inputs})[0]
        with tf.variable_scope('predict'):
            reconsturct_error = self.sess_ssm.run(
                self.reconstruct_error,
                feed_dict={
                    self.input: inputs, self.label:inputs
                }
            )
        return reconsturct_error  # use the reconstruction loss as the penalty

    def build(self):
        '''
        Build the computation graph
        '''
        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self._predict()
        self.summary()
        # initialize variables
        self.sess_ssm.run(tf.global_variables_initializer())

    def train(self,
              inputs,
              targets,
              batch_size=256,
              max_epochs=100,
              holdout_ratio=0.2,
              max_grad_updates=None,
              max_logging=1000,  # TODO: what is this used for?
              hide_progress=False,
              max_t=None):
        """
        @params: inputs: (s, a, s') pair. shape [#buffer, 2*obs_dim+act_dim]
        """

        self._state = {}
        break_train = False  # used to break the training once the holdout_losses doesn't improve
        #

        # split into training and holdout sets
        num_holdout = min(int(inputs.shape[0] * holdout_ratio), max_logging)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, holdout_inputs = inputs[permutation[num_holdout:]], inputs[permutation[:num_holdout]]
        targets, holdout_targets = targets[permutation[num_holdout:]], targets[permutation[:num_holdout]]
        # expand the inputs data to fit the shape of nn
        # holdout_inputs = np.tile(holdout_inputs[None], [self.num_nets, 1, 1])

        # print('[ SSM ] Training{} | Holdout: {}'.format(inputs.shape, holdout_inputs.shape))

        idxs = np.arange(inputs.shape[0])

        if hide_progress:
            progress = Silent()
        else:
            progress = Progress(max_epochs)

        grad_updates = 0
        val_loss = 0
        if max_epochs is not None:
            epoch_iter = range(max_epochs)
        else:
            epoch_iter = itertools.count()

        # epoch_iter = range(1)
        for epoch in epoch_iter:
            for batch_num in range(int(np.ceil(idxs.shape[-1] / batch_size))):
                batch_idxs = idxs[batch_num * batch_size:(batch_num + 1) * batch_size]
                self.sess_ssm.run(
                    self.optimizer,
                    feed_dict={self.input: inputs[batch_idxs],
                               self.label: targets[batch_idxs]}
                )
                # print("check vaule when training, logp: {}, grad1: {}, gradv: {}, grad2: {}, loss1:{}, loss2:{}, loss:{}".format(
                #     logp_o, grad1_o, gradv_o, grad2_o, loss1_o, loss2_o, loss_o
                # ))
                grad_updates += 1

            # shuffle data for eval
            # idxs = shuffle_rows(idxs)
            np.random.shuffle(idxs)

            # val and logging
            if not hide_progress:
                model_loss = self.eval(inputs[idxs[:max_logging]], targets[idxs[:max_logging]])
                holdout_loss = self.eval(holdout_inputs, holdout_targets)

                wandb.log({'AE': {'val_loss': holdout_loss, 'train_loss': model_loss}})

                # for printing
                named_losses = [['M', model_loss]]
                named_holdout_losses = [['V', holdout_loss]]
                named_losses = named_losses + named_holdout_losses
                progress.set_description(named_losses)

                break_train = self._save_best(epoch, holdout_loss)

            progress.update()

            # stop training
            if break_train or (max_grad_updates and grad_updates > max_grad_updates):
                break

        val_loss = self.eval(holdout_inputs, holdout_targets)
        print(' [ AE ] Finish training, the validation loss is :', val_loss)

        return OrderedDict({'val_loss': val_loss})

    def _save_best(self, epoch, holdout_loss):
        """
        save the checkpoint and early stop
        The condition of early stopping: (best - current)/current > 0.01 and

        """
        updated = False

        current = holdout_loss
        _, best = self._snapshot
        improvement = (best - current) / best  # Notice this is different with the one used in bnn._save_best
        print("improvement {} and updates steps {} and current holdout_loss {}, best loss {}".format(improvement,
                                                                                                     self._epochs_since_update,
                                                                                                     current, best))
        if improvement > 0.01:
            self._snapshot = (epoch, current)
            # save current state
            # saver.save(self.sess_ssm, '')
            updated = True

        # early stopping
        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1

        if self._epochs_since_update > self._early_stop_patience:
            return True
        else:
            return False

