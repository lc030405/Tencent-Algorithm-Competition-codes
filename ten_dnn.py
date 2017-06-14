from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import zipfile

from six.moves import urllib

import pandas as pd
import tensorflow as tf
import  numpy as np

COLUMNS = ['clickTime', 'connectionType', 'telecomsOperator', 'adID',
           'camgaignID', 'advertiserID', 'appID', 'appPlatform',
           'age', 'gender', 'education', 'marriageStatus',
           'haveBaby', 'hometown', 'residence', 'sitesetID',
           'positionType']
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["connectionType", "telecomsOperator", "adID", "camgaignID", "advertiserID",
                       'appID', 'appPlatform', 'gender', 'marriageStatus',
                       'hometown', 'residence', 'sitesetID', 'positionType']
CONTINUOUS_COLUMNS = ["clickTime", "age", "education", "haveBaby"]


def build_estimator(model_dir, model_type):
    """Build an estimator."""
    """
    CATEGORICAL_COLUMNS = ["creativeID", "userID", "positionID", "connectionType",
                         "telecomsOperator", "adID", "camgaignID", "advertiserID",
                         'appID', 'appPlatform','gender','marriageStatus',
                         'hometown','residence','sitesetID', 'positionType']
    """
    # Sparse base columns.

    connectionType = tf.contrib.layers.sparse_column_with_hash_bucket(
        "connectionType", hash_bucket_size=10)
    telecomsOperator = tf.contrib.layers.sparse_column_with_hash_bucket(
        "telecomsOperator", hash_bucket_size=10)
    adID = tf.contrib.layers.sparse_column_with_hash_bucket(
        "adID", hash_bucket_size=1e5)
    camgaignID = tf.contrib.layers.sparse_column_with_hash_bucket(
        "camgaignID", hash_bucket_size=1e5)
    advertiserID = tf.contrib.layers.sparse_column_with_hash_bucket(
        "advertiserID", hash_bucket_size=1e5)
    appID = tf.contrib.layers.sparse_column_with_hash_bucket(
        "appID", hash_bucket_size=1e5)
    appPlatform = tf.contrib.layers.sparse_column_with_hash_bucket(
        "appPlatform", hash_bucket_size=10)
    gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender", keys=["0", "1", "2"])
    marriageStatus = tf.contrib.layers.sparse_column_with_hash_bucket(
        "marriageStatus", hash_bucket_size=10)
    hometown = tf.contrib.layers.sparse_column_with_hash_bucket(
        "hometown", hash_bucket_size=1e5)
    residence = tf.contrib.layers.sparse_column_with_hash_bucket(
        "residence", hash_bucket_size=1e5)
    sitesetID = tf.contrib.layers.sparse_column_with_hash_bucket(
        "sitesetID", hash_bucket_size=10)
    positionType = tf.contrib.layers.sparse_column_with_hash_bucket(
        "positionType", hash_bucket_size=10)

    """
    CONTINUOUS_COLUMNS = ["clickTime", "age", "education", "haveBaby"]
    """

    # Continuous base columns.
    clickTime = tf.contrib.layers.real_valued_column("clickTime")
    age = tf.contrib.layers.real_valued_column("age")
    education = tf.contrib.layers.real_valued_column("education")
    haveBaby = tf.contrib.layers.real_valued_column("haveBaby")

    # Transformations.
    age_buckets = tf.contrib.layers.bucketized_column(age,
                                                      boundaries=[
                                                          18, 25, 30, 35, 40, 45,
                                                          50, 55, 60, 65
                                                      ])
    clickTime_buckets = tf.contrib.layers.bucketized_column( clickTime,
                                                             boundaries = [
                                                                 170001, 170010, 170020, 170030,
                                                                 170040, 170050, 170060, 170070,
                                                                 170080, 170090, 170100, 170110,
                                                                 170120, 170130, 170140, 170150,
                                                                 170160, 170170, 170180, 170190,
                                                                 170200, 170210, 170220, 170230,
                                                                 170280, 170330, 170380, 170420
                                                             ])

    # Wide columns and deep columns.
    wide_columns = [clickTime_buckets, connectionType, telecomsOperator, adID, age_buckets, camgaignID, advertiserID,
                    appID, appPlatform, marriageStatus, hometown, residence, sitesetID,
                    positionType,
                    tf.contrib.layers.crossed_column([marriageStatus, gender],
                                                     hash_bucket_size=int(1e2)),
                    tf.contrib.layers.crossed_column([hometown, residence],
                                                     hash_bucket_size=int(1e6)),
                    tf.contrib.layers.crossed_column([sitesetID, positionType],
                                                     hash_bucket_size=int(1e6)),
                    tf.contrib.layers.crossed_column([appPlatform, appID],
                                                     hash_bucket_size=int(1e6)),
                    tf.contrib.layers.crossed_column([appPlatform, adID],
                                                     hash_bucket_size=int(1e6)),
                    tf.contrib.layers.crossed_column([appPlatform, advertiserID],
                                                     hash_bucket_size=int(1e6))
                    ]

    deep_columns = [
        tf.contrib.layers.embedding_column(connectionType, dimension=3),
        tf.contrib.layers.embedding_column(telecomsOperator, dimension=3),
        tf.contrib.layers.embedding_column(gender, dimension=3),
        tf.contrib.layers.embedding_column(marriageStatus, dimension=3),
        # connectionType,
        # telecomsOperator,
        # gender,
        # marriageStatus,
        haveBaby,
        education,
        age
    ]

    if model_type == "wide":
        m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                              feature_columns=wide_columns)
    elif model_type == "deep":
        m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                           feature_columns=deep_columns,
                                           hidden_units=[100, 50])
    else:
        m = tf.contrib.learn.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[100, 50],
            fix_global_step_increment_bug=True)
    return m


def input_fn(df, train=False):
    """Input builder function."""
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {
        k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS}
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)
    # Converts the label column into a constant Tensor.
    if train:
        label = tf.constant(df[LABEL_COLUMN].values)
        # Returns the feature columns and the label.
        return feature_cols, label
    else:
        return feature_cols


def train_and_eval(model_dir, model_type, train_steps, train_data, test_data):
    # load data
    data_root = "pre"
    df_train = pd.read_csv("%s/train.csv" % data_root, dtype={'connectionType': str, 'telecomsOperator': str})
    df_test = pd.read_csv("%s/test.csv" % data_root, dtype={'connectionType': str, 'telecomsOperator': str})
    dfAd = pd.read_csv("%s/ad.csv" % data_root, dtype={'adID': str, 'camgaignID': str, 'advertiserID': str,
                                                       'appID': str, 'appPlatform': str})
    dfUser = pd.read_csv("%s/user.csv" % data_root, dtype={'gender': str, 'marriageStatus': str,
                                                           'hometown': str, 'residence': str})
    dfPosition = pd.read_csv("%s/position.csv" % data_root, dtype={'sitesetID': str, 'positionType': str})

    # process data
    df_train = pd.merge(df_train, dfAd, on="creativeID")
    df_train = pd.merge(df_train, dfUser, on="userID")
    df_train = pd.merge(df_train, dfPosition, on="positionID")

    df_test = pd.merge(df_test, dfAd, on="creativeID")
    df_test = pd.merge(df_test, dfUser, on="userID")
    df_test = pd.merge(df_test, dfPosition, on="positionID")

    # remove NaN elements
    df_train = df_train.dropna(how='any', axis=0)
    df_test = df_test.dropna(how='any', axis=0)
    df_train.drop(['conversionTime', 'userID', 'creativeID', 'positionID'], axis=1, inplace=True)
    df_test.drop(['userID', 'creativeID', 'positionID'], axis=1, inplace=True)

    model_dir = tempfile.mkdtemp() if not model_dir else model_dir
    print("model directory = %s" % model_dir)

    m = build_estimator(model_dir, model_type)

    m.fit(input_fn=lambda: input_fn(df_train, True), steps=train_steps)
    result = m.predict_proba(input_fn=lambda: input_fn(df_test, False))
    proba = []
    for value in result:
        # for elem in value:
        # print(value, ' ')
        # print(type(value))
        # print(type(value[0]))
        proba.append(value[0])

    # submission
    df = pd.DataFrame({"instanceID": df_test["instanceID"].values, "proba": np.array(proba)})
    df.sort_values("instanceID", inplace=True)
    df.to_csv("submission.csv", index=False)
    with zipfile.ZipFile("submission.zip", "w") as fout:
        fout.write("submission.csv", compress_type=zipfile.ZIP_DEFLATED)

FLAGS = None


def main(_):
    train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                   FLAGS.train_data, FLAGS.test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="Base directory for output models."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="wide_n_deep",
        help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=200,
        help="Number of training steps."
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="",
        help="Path to the training data."
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="",
        help="Path to the test data."
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
