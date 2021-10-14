"""Unit Test for TransformLayer"""
import pandas as pd
from faker import Faker
import tensorflow as tf
from datatransformer import tensorflow as dtf
from modelbricks.layers.layers import TransformLayer

class TestTransformLayer(tf.test.TestCase):
    """Test Case for Transform Layer"""
    def setUp(self):
        super().setUp()

        #Fake Data
        myfaker = Faker()
        fake_dim1_data = [
                [
                    myfaker.random_int(), myfaker.country_code(), str(myfaker.date_time()),
                    myfaker.pyfloat(), str(myfaker.pybool()), myfaker.word()
                ] for i in range(10)
        ]
        fake_dim1 = pd.DataFrame(
            fake_dim1_data, columns = [f'f_feature_{i}' for i in range(6)]
        )

        fake_dim2_data = [
                [
                    myfaker.random_int(min=1553,max=1563,step=1),
                    str(myfaker.date_time()), str(myfaker.pybool()),
                    myfaker.word(), myfaker.bothify(text='?'),
                    myfaker.pyfloat()
                ] for i in range(20)
        ]
        fake_dim2 = pd.DataFrame(
            fake_dim2_data, columns = [f'b_feature_{i}' if i!=0 else 'trans_id' for i in range(6)]
        )

        self.feature_columns = {
            "foo": {
                "f_feature_0": [0, 1, 2, 3, 4],
                "f_feature_1": [
                    "JP", "HK", "IN", "ID", "NL",
                    "KH", "CN", "TH", "MY", "VN"
                ],
                "f_feature_2": [
                    "2021-08-08 08:15:54.550", "2021-07-30 03:45:32.517",
                    "2021-08-18 11:49:47.460", "2021-07-07 12:05:45.600",
                    "2021-08-23 10:53:20.270", "2021-08-02 10:53:11.160",
                    "2021-07-16 12:54:44.997"
                ],
                "f_feature_4": ["False", "True"],
                "f_feature_5": ["person", "station", "song"],
            },
            "bar": {
                "b_feature_1": [
                    "2021-08-08 08:15:54.550", "2021-07-30 03:45:32.517",
                    "2021-08-18 11:49:47.460", "2021-07-07 12:05:45.600",
                    "2021-08-23 10:53:20.270", "2021-08-02 10:53:11.160",
                    "2021-07-16 12:54:44.997"
                ],
                "b_feature_2": ["False", "True"],
                "b_feature_3": ["person", "station", "song"],
                "b_feature_4": [
                    "a", "R", "c", "j", "L",
                    "p", "K", "h", "N", "v"
                ],
            }
        }

        self.data = {'foo': fake_dim1, 'bar': fake_dim2}
        self.dataspec={
            'foo': {
                'type': 'non_sequential',
                'sparse_feature': [
                    'f_feature_0', 'f_feature_1', 'f_feature_2',
                    'f_feature_4', 'f_feature_5',
                ],
                'dense_feature': [
                    'f_feature_3',
                ],
            },
            'bar': {
                'type': 'sequential',
                'sparse_feature': [
                    'b_feature_1', 'b_feature_2', 'b_feature_3', 'b_feature_4'
                ],
                'dense_feature': [
                    'b_feature_5',
                ],
            },
        }

        datatrans = dtf.TensorflowDataTransformer(
            data = self.data,
            data_spec = self.dataspec,
            feature_column_config = self.feature_columns
        )

        self.dataset = datatrans.to_dataset()
        self.feature_columns = datatrans.feature_columns
        self.dim = {0: 'non_sequential', 1:'sequential'}

        for inputs in self.dataset.take(1):
            self.trans_from = TransformLayer(self.feature_columns, self.dim)
            self.trans_from.build(inputs)

    def testdatasetelementspec(self):
        """Test data input element spec"""
        excepted_ele_spec = (
            {
                'f_feature_0': tf.TensorSpec(
                    shape=(None,), dtype=tf.int64, name=None
                ),
                'f_feature_1': tf.TensorSpec(
                    shape=(None,), dtype=tf.string, name=None
                ),
                'f_feature_2': tf.TensorSpec(
                    shape=(None,), dtype=tf.string, name=None
                ),
                'f_feature_4': tf.TensorSpec(
                    shape=(None,), dtype=tf.string, name=None
                ),
                'f_feature_5': tf.TensorSpec(
                    shape=(None,), dtype=tf.string, name=None
                ),
                'f_feature_3': tf.TensorSpec(
                    shape=(None,), dtype=tf.float64, name=None
                )
            },
            {
                'b_feature_1': tf.RaggedTensorSpec(
                    tf.TensorShape([None, None, None]), tf.string, 2, tf.int64
                ),
                'b_feature_2': tf.RaggedTensorSpec(
                    tf.TensorShape([None, None, None]), tf.string, 2, tf.int64
                ),
                'b_feature_3': tf.RaggedTensorSpec(
                    tf.TensorShape([None, None, None]), tf.string, 2, tf.int64
                ),
                'b_feature_4': tf.RaggedTensorSpec(
                    tf.TensorShape([None, None, None]), tf.string, 2, tf.int64
                ),
                'b_feature_5': tf.RaggedTensorSpec(
                    tf.TensorShape([None, None, None]), tf.float32, 2, tf.int64
                )
            }
        )

        self.assertEqual(excepted_ele_spec, self.dataset.element_spec)

    def testdatasetfeaturecolumns(self):
        """Test feature column of datatransformer"""
        excepted_feature_columns={
            'non_sequential': {
                'foo': {
                    'dense': [
                        tf.feature_column.numeric_column(
                            key='f_feature_3', shape=(1,),
                            default_value=None, dtype=tf.float32, normalizer_fn=None
                        )],
                    'sparse': [
                        tf.feature_column.categorical_column_with_vocabulary_list(
                            key='f_feature_0', vocabulary_list=(0, 1, 2, 3, 4),
                            dtype=tf.int64, default_value=-1, num_oov_buckets=0
                        ),
                        tf.feature_column.categorical_column_with_vocabulary_list(
                            key='f_feature_1',
                            vocabulary_list=(
                                'JP', 'HK', 'IN', 'ID', 'NL',
                                'KH', 'CN', 'TH', 'MY', 'VN'
                            ),
                            dtype=tf.string, default_value=-1, num_oov_buckets=0
                        ),
                        tf.feature_column.categorical_column_with_vocabulary_list(
                            key='f_feature_2',
                            vocabulary_list=(
                                '2021-08-08 08:15:54.550', '2021-07-30 03:45:32.517',
                                '2021-08-18 11:49:47.460', '2021-07-07 12:05:45.600',
                                '2021-08-23 10:53:20.270', '2021-08-02 10:53:11.160',
                                '2021-07-16 12:54:44.997'
                            ),
                            dtype=tf.string, default_value=-1, num_oov_buckets=0
                        ),
                        tf.feature_column.categorical_column_with_vocabulary_list(
                            key='f_feature_4', vocabulary_list=('False', 'True'),
                            dtype=tf.string, default_value=-1, num_oov_buckets=0
                        ),
                        tf.feature_column.categorical_column_with_vocabulary_list(
                            key='f_feature_5', vocabulary_list=('person', 'station', 'song'),
                            dtype=tf.string, default_value=-1, num_oov_buckets=0
                        )
                    ]
                }
            },
            'sequential': {
                'bar': {
                    'dense': [
                        tf.feature_column.sequence_numeric_column(
                            key='b_feature_5', shape=(1,), default_value=0.0,
                            dtype=tf.float32, normalizer_fn=None
                        ),
                    ],
                    'sparse': [
                        tf.feature_column.sequence_categorical_column_with_vocabulary_list(
                            key='b_feature_1',
                            vocabulary_list=(
                                '2021-08-08 08:15:54.550', '2021-07-30 03:45:32.517',
                                '2021-08-18 11:49:47.460', '2021-07-07 12:05:45.600',
                                '2021-08-23 10:53:20.270', '2021-08-02 10:53:11.160',
                                '2021-07-16 12:54:44.997'
                            ),
                            dtype=tf.string, default_value=-1, num_oov_buckets=0
                        ),
                        tf.feature_column.sequence_categorical_column_with_vocabulary_list(
                            key='b_feature_2', vocabulary_list=('False', 'True'),
                            dtype=tf.string, default_value=-1, num_oov_buckets=0
                        ),
                        tf.feature_column.sequence_categorical_column_with_vocabulary_list(
                            key='b_feature_3', vocabulary_list=('person', 'station', 'song'),
                            dtype=tf.string, default_value=-1, num_oov_buckets=0
                        ),
                        tf.feature_column.sequence_categorical_column_with_vocabulary_list(
                            key='b_feature_4',
                            vocabulary_list=('a', 'R', 'c', 'j', 'L', 'p', 'K', 'h', 'N', 'v'),
                            dtype=tf.string, default_value=-1, num_oov_buckets=0
                        )
                    ]
             }
            }
        }
        self.assertEqual(excepted_feature_columns, self.feature_columns)

    def testtransformlayeroutputshape(self):
        """test Transformer Layer output shape"""
        for input_x in self.dataset.take(1):
            output = self.trans_from(input_x)

        expected_output_shape = tf.TensorShape([1,38])

        self.assertAllEqual(expected_output_shape, output.shape)

tf.test.main()
