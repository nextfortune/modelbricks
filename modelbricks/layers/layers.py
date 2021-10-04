import tensorflow as tf

class TransformLayer(tf.keras.layers.Layer):
    """Creates a Transform Layers."""
    def __init__(self, features_columns, dim_type):
        """Creates a Transform Layers.
        Args:
          features_columns: A dictionary that store tensorflow features columns
                           (like numeric_column or categorical_column_with_vocabulary_list)
          dim_type: A dictionary to show which dim is non_sequential or sequential
          ```python
          dim_type = {0: 'non_sequential', 1:'sequential'}
          features_columns = {
              'league_id' = [tf.feature_column.categorical_column_with_vocabulary_list],
              'odds' = [feature_column.numeric_column],
          }
        """
        super(TransformLayer, self).__init__()
        self.dim_type = dim_type

        self.features_layers = {}
        for dim in features_columns['non_sequential'].keys():
            for dcat in features_columns['non_sequential'][dim].keys():
                if dcat =='sparse':
                    for col in range(len(features_columns['non_sequential'][dim][dcat])):
                        key = features_columns['non_sequential'][dim][dcat][col].name
                        embedded = tf.feature_column.embedding_column(
                            features_columns['non_sequential'][dim][dcat][col], dimension=4
                        )
                        self.features_layers[key] = tf.keras.layers.DenseFeatures(
                            embedded, trainable=True
                        )
                else:
                    for col in range(len(features_columns['non_sequential'][dim][dcat])):
                        key = features_columns['non_sequential'][dim][dcat][col].name
                        self.features_layers[key] = tf.keras.layers.DenseFeatures(
                            features_columns['non_sequential'][dim][dcat][col], trainable=True
                        )

        for dim in features_columns['sequential'].keys():
            for dcat in features_columns['sequential'][dim].keys():
                if dcat =='sparse':
                    for col in range(len(features_columns['sequential'][dim][dcat])):
                        key = features_columns['sequential'][dim][dcat][col].name
                        embedded = tf.feature_column.embedding_column(
                            features_columns['sequential'][dim][dcat][col], dimension=4
                        )
                        self.features_layers[key] = tf.keras.experimental.SequenceFeatures(
                            embedded, trainable=True
                        )
                else:
                    for col in range(len(features_columns['sequential'][dim][dcat])):
                        key = features_columns['sequential'][dim][dcat][col].name
                        self.features_layers[key] = tf.keras.experimental.SequenceFeatures(
                            features_columns['sequential'][dim][dcat][col], trainable=True
                        )

    def call(self, inputs):
        trainable_dic = {}
        for dim_type in self.dim_type.keys():
            if self.dim_type[dim_type] == 'non_sequential':
                fea_ten_dic = {
                    key: self.features_layers[key]({key: inputs[dim_type][key]})
                    for key in sorted(inputs[dim_type].keys())
                }
                trainable_dic[dim_type] = tf.concat(
                    [fea_ten_dic[key] for key in sorted(fea_ten_dic.keys())], axis=1
                )
            else:
                fea_ten_dic = {
                    key: tf.reduce_mean(
                        self.features_layers[key]({key: inputs[dim_type][key].to_sparse()})[0],
                        axis=1
                    )
                    for key in sorted(inputs[dim_type].keys())
                }
                trainable_dic[dim_type] = tf.concat(
                    [fea_ten_dic[key] for key in sorted(fea_ten_dic.keys())],axis=1
                )

        trainable = tf.concat([trainable_dic[key] for key in sorted(trainable_dic.keys())],axis=1)

        return trainable
