"""Unit Test for Rucmodel"""
import pandas as pd
from faker import Faker
import tensorflow as tf
from datatransformer import tensorflow as dtf
from modelbricks.models.models import RucModel
from modelbricks.metrics.metrics import F1Score

class Testrucmodel(tf.test.TestCase):
    """Test Rucmodel test case"""
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

        fake_label_data = [myfaker.pybool() for i in range(10)]
        fake_label = pd.DataFrame(fake_label_data, columns=['output_1'])

        feature_columns = {
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

        data = {'foo': fake_dim1, 'bar': fake_dim2, 'labels': fake_label}
        dataspec={
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
            'labels': {
                'type': 'non_sequential',
                'label': ['output_1']
            }
        }

        datatrans = dtf.TensorflowDataTransformer(
            data = data,
            data_spec = dataspec,
            feature_column_config = feature_columns
        )

        self.dataset = datatrans.to_dataset()
        self.features_columns = datatrans.feature_columns
        self.dim = {0: 'non_sequential', 1:'sequential'}
        self.label = 'bad'

        self.model = RucModel(self.features_columns, self.dim, self.label)
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

    def testrucmodeloutputshape(self):
        """test model output shape"""
        #pylint: disable=W0612
        for (trans, games), lables in self.dataset.take(1):
            input_x = (trans, games)
            output = self.model.call(input_x)

        expected_output_shape = tf.TensorShape([1,1])

        self.assertEqual(expected_output_shape, output.shape)

    def testtrainingweightschange(self):
        """test model weights changes after training"""
        train_dataset = self.dataset.take(10).batch(10)
        before_weights = self.model.weights
        self.model.fit(train_dataset, epochs = 1)
        after_weights = self.model.weights

        self.assertNotEqual(before_weights, after_weights)

tf.test.main()
