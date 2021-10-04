import tensorflow as tf
from modelbricks.layers.layers import TransformLayer
from modelbricks.metrics.metrics import F1Score

class RucModel(tf.keras.models.Model):
    """Creates a RiskUserClassification Model."""
    def __init__(self, features_columns, dim_type, label):
        """Creates a RiskUserClassification Model.
        Args:
          features_columns: A dictionary that store tensorflow features columns
                           (like numeric_column or categorical_column_with_vocabulary_list)
          dim_type: A dictionary to show which dim is non_sequential or sequential
          label: The label column name in our data

        '''python
        dim_type = {0: 'non_sequential', 1:'sequential'}
        features_columns = {
              'league_id' = [tf.feature_column.categorical_column_with_vocabulary_list],
              'odds' = [feature_column.numeric_column],
        }
        label = 'bad'

        ruc = RucModel(features_columns, dim_type, label)
        """
        super(RucModel, self).__init__()
        self.trans_layer = TransformLayer(features_columns, dim_type)
        self.dense_layer = tf.keras.layers.Dense(34, activation='relu')
        self.dense_layer2 = tf.keras.layers.Dense(15, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        self.label = label

    @tf.function
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        x, y = data

        y_pred = self(x, training=False)
        loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def get_config(self):

        return {'accuracy': tf.keras.metrics.Accuracy(),
                'Recall': tf.keras.metrics.Recall(),
                'Precision': tf.keras.metrics.Precision(),
                'TP': tf.keras.metrics.TruePositives(),
                'F1_Score': F1Score(),}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        trainable = self.trans_layer(inputs)
        dly1_out = self.dense_layer(trainable)
        dly2_out = self.dense_layer2(dly1_out)
        out = self.output_layer(dly2_out)

        return out
