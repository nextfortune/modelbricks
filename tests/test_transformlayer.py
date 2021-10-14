"""Unit Test for TransformLayer"""
import tensorflow as tf
from modelbricks.layers.layers import TransformLayer

import generate_fake_data
import common_base_test as cbt

class TestTransformLayer(cbt.TestBase):
    """Test Case for Transform Layer"""

    def setUp(self):
        super().setUp()

        datatrans = generate_fake_data.generate_fake_data()
        self.dataset = datatrans.to_dataset()
        self.feature_columns = datatrans.feature_columns
        self.dim = {0: 'non_sequential', 1:'sequential'}

        #pylint: disable=W0612
        for inputs, labels in self.dataset.take(1):
            self.trans_from = TransformLayer(self.feature_columns, self.dim)
            self.trans_from.build(inputs)

    def testtransformlayeroutputshape(self):
        """test Transformer Layer output shape"""
        #pylint: disable=W0612
        for input_x, labels in self.dataset.take(1):
            output = self.trans_from(input_x)

        expected_output_shape = tf.TensorShape([1,38])

        self.assertAllEqual(expected_output_shape, output.shape)

tf.test.main()
