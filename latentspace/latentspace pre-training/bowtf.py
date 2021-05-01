import numpy
import tensorflow as tf
import pandas as pd
import math


class BOWTF:

    def __init__(self, inputnodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.activation_function = lambda x: numpy.tanh(x)

        with tf.device("/device:gpu:0"):
            self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)))

        self.matrix = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.inodes)).tolist()

        self.matrix_holder = tf.placeholder(tf.float32)
        self.input_holder = tf.placeholder(tf.float32)
        self.final_inputs_holder = tf.matmul(self.matrix_holder, self.input_holder)
        self.targets_holder = tf.placeholder(tf.float32)
        self.final_outouts_holder = tf.placeholder(tf.float32)
        self.output_errors_holder = self.targets_holder - self.final_outouts_holder
        self.error_delta_holder = tf.placeholder(tf.float32)
        self.inputs_T_holder = tf.placeholder(tf.float32)
        self.matrix_update_holder = self.matrix_holder + self.lr * tf.matmul(self.error_delta_holder, self.inputs_T_holder)

        self.errors = []

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).tolist()
        inputs_T = numpy.array(inputs_list, ndmin=2).T
        inputs_T = inputs_T.tolist()
        targets = numpy.array(targets_list, ndmin=2).T
        targets = targets.tolist()

        final_inputs = self.sess.run(self.final_inputs_holder, {self.matrix_holder: self.matrix, self.input_holder: inputs_T})
        final_outputs = self.activation_function(final_inputs)

        output_errors = self.sess.run(self.output_errors_holder, {self.targets_holder: targets, self.final_outouts_holder: final_outputs})
        self.errors.append(output_errors)

        error_delta = output_errors * (1 - final_outputs * final_outputs)
        self.matrix = self.sess.run(self.matrix_update_holder, {self.matrix_holder: self.matrix, self.error_delta_holder: error_delta, self.inputs_T_holder: inputs})



    def save_matrix_to_csv(self, filename):
        dataframe = pd.DataFrame(data=self.matrix.astype(float))
        dataframe.to_csv(filename, sep='*', header=False, index=False)


    def error_avg(self):
        with tf.device("/gpu:0"):
            with tf.Session() as sess:
                error_sum_i = tf.constant([0.0 for i in range(self.onodes)])
                error_sum = sess.run(tf.reshape(error_sum_i, [self.onodes, 1]))
                for error in self.errors:
                    error_sum += error
                error_avg = error_sum / len(self.errors)
                error_value_square = 0
                for i in error_avg:
                    tmp = i * i
                    error_value_square += tmp
                error_value = math.sqrt(error_value_square)
                self.errors = []
                return error_value




