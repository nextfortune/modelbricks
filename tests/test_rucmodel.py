"""Unit Test for Rucmodel"""
import unittest
from modelbricks.models.models import RucModel
from modelbricks.metrics.metrics import F1Score

import tensorflow as tf
import common_base_test as cbt

class Testrucmodel(cbt.TestBase):
    """Test Rucmodel test case"""
    def setUp(self):
        super().setUp()

        self.dim = {0: 'non_sequential', 1:'sequential'}
        self.label = 'output_1'

        self.model = RucModel(self.feature_columns, self.dim, self.label)
        loss_object = tf.keras.losses.BinaryCrossentropy()
        optimizer_adae = tf.keras.optimizers.Adadelta(learning_rate=1.0,rho=0.90)
        #model compile
        self.model.compile(
            optimizer=optimizer_adae,
            loss=loss_object,
            metrics=[
                'accuracy', 'Recall', 'Precision',
                tf.keras.metrics.TruePositives(), F1Score()
            ]
        )

    def test_rucmodel_outputshape(self):
        """test model output shape"""
        #pylint: disable=W0612
        for (trans, games), lables in self.dataset.take(1):
            input_x = (trans, games)
            output = self.model.call(input_x)

        expected_output_shape = tf.TensorShape([1,1])

        self.assertEqual(expected_output_shape, output.shape)

    def test_training_weights_change(self):
        """test model weights changes after training"""
        train_dataset = self.dataset.take(10).batch(10)
        before_weights = self.model.weights
        hist = self.model.fit(train_dataset, epochs = 1, verbose=0).history
        after_weights = self.model.weights

        self.assertNotEqual(before_weights, after_weights)
