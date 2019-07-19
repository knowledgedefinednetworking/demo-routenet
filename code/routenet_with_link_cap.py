# Copyright (c) 2019, Krzysztof Rusek [^1], Paul Almasan [^2]
#
# [^1]: AGH University of Science and Technology, Department of
#     communications, Krakow, Poland. Email: krusek\@agh.edu.pl
#
# [^2]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: almasan@ac.upc.edu


from __future__ import print_function

import numpy as np
import pandas as pd
import networkx as nx
import itertools as it
import os
import tensorflow as tf
from tensorflow import keras
import re
import argparse
import random
import tarfile
import glob

def genPath(R,s,d,connections):
    while s != d:
        yield s
        s = connections[s][R[s,d]]
    yield s

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)

def load_routing(routing_file):
    R = pd.read_csv(routing_file, header=None, index_col=False)
    R=R.drop([R.shape[0]], axis=1)
    return R.values

def make_indices(paths):
    link_indices=[]
    path_indices=[]
    sequ_indices=[]
    segment=0
    for p in paths:
        link_indices += p
        path_indices += len(p)*[segment]
        sequ_indices += list(range(len(p)))
        segment +=1
    return link_indices, path_indices, sequ_indices

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_features(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def parse(serialized, target='delay'):
    '''
    Target is the name of predicted variable
    '''
    with tf.device("/cpu:0"):
        with tf.name_scope('parse'):
            features = tf.parse_single_example(
                serialized,
                features={
                    'traffic':tf.VarLenFeature(tf.float32),
                    target:tf.VarLenFeature(tf.float32),
                    'link_capacity': tf.VarLenFeature(tf.float32),
                    'links':tf.VarLenFeature(tf.int64),
                    'paths':tf.VarLenFeature(tf.int64),
                    'sequences':tf.VarLenFeature(tf.int64),
                    'n_links':tf.FixedLenFeature([],tf.int64),
                    'n_paths':tf.FixedLenFeature([],tf.int64),
                    'n_total':tf.FixedLenFeature([],tf.int64)
                })
            for k in ['traffic',target,'link_capacity','links','paths','sequences']:
                features[k] = tf.sparse_tensor_to_dense(features[k])
                if k == 'delay':
                    features[k] = (features[k] - 0.37) / 0.54
                if k == 'traffic':
                    features[k] = (features[k] - 0.17) / 0.13
                if k == 'link_capacity':
                    features[k] = (features[k] - 25.0) / 40.0



    return {k:v for k,v in features.items() if k is not target },features[target]

def cummax(alist, extractor):
    with tf.name_scope('cummax'):
        maxes = [tf.reduce_max( extractor(v) ) + 1 for v in alist ]
        cummaxes = [tf.zeros_like(maxes[0])]
        for i in range(len(maxes)-1):
            cummaxes.append( tf.math.add_n(maxes[0:i+1]))

    return cummaxes


def transformation_func(it, batch_size=32):
    with tf.name_scope("transformation_func"):
        vs = [it.get_next() for _ in range(batch_size)]

        links_cummax = cummax(vs,lambda v:v[0]['links'] )
        paths_cummax = cummax(vs,lambda v:v[0]['paths'] )

        tensors = ({
                'traffic':tf.concat([v[0]['traffic'] for v in vs], axis=0),
                'sequences':tf.concat([v[0]['sequences'] for v in vs], axis=0),
                'link_capacity': tf.concat([v[0]['link_capacity'] for v in vs], axis=0),
                'links':tf.concat([v[0]['links'] + m for v,m in zip(vs, links_cummax) ], axis=0),
                'paths':tf.concat([v[0]['paths'] + m for v,m in zip(vs, paths_cummax) ], axis=0),
                'n_links':tf.math.add_n([v[0]['n_links'] for v in vs]),
                'n_paths':tf.math.add_n([v[0]['n_paths'] for v in vs]),
                'n_total':tf.math.add_n([v[0]['n_total'] for v in vs])
            },   tf.concat([v[1] for v in vs], axis=0))

    return tensors

def tfrecord_input_fn(filenames,hparams,shuffle_buf=1000,target='delay'):

    files = tf.data.Dataset.from_tensor_slices(filenames)
    files = files.shuffle(len(filenames))

    ds = files.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=4))

    if shuffle_buf:
        ds = ds.apply(tf.contrib.data.shuffle_and_repeat(shuffle_buf))

    ds = ds.map(lambda buf:parse(buf,target),
        num_parallel_calls=2)
    ds=ds.prefetch(10)

    it =ds.make_one_shot_iterator()
    sample = transformation_func(it,hparams.batch_size)


    return sample

class ComnetModel(tf.keras.Model):
    def __init__(self,hparams, output_units=1, final_activation=None):
        super(ComnetModel, self).__init__()
        self.hparams = hparams

        self.edge_update = tf.keras.layers.GRUCell(hparams.link_state_dim)
        self.path_update = tf.keras.layers.GRUCell(hparams.path_state_dim)


        self.readout = tf.keras.models.Sequential()

        self.readout.add(keras.layers.Dense(hparams.readout_units,
                activation=tf.nn.selu,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(hparams.l2)))
        self.readout.add(keras.layers.Dropout(rate=hparams.dropout_rate))
        self.readout.add(keras.layers.Dense(hparams.readout_units,
                activation=tf.nn.selu,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(hparams.l2)))
        self.readout.add(keras.layers.Dropout(rate=hparams.dropout_rate))

        self.readout.add(keras.layers.Dense(output_units,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(hparams.l2_2),
                activation = final_activation ) )


    def build(self, input_shape=None):
        del input_shape
        self.edge_update.build(tf.TensorShape([None,self.hparams.path_state_dim]))
        self.path_update.build(tf.TensorShape([None,self.hparams.link_state_dim]))
        self.readout.build(input_shape = [None,self.hparams.path_state_dim])
        self.built = True


    def call(self, inputs, training=False):
        f_ = inputs
        shape = tf.stack([f_['n_links'], self.hparams.link_state_dim-1], axis=0)
        link_state = tf.concat([
            tf.expand_dims(f_['link_capacity'], axis=1),
            tf.zeros(shape)
        ], axis=1)
        shape = tf.stack([f_['n_paths'],self.hparams.path_state_dim-1], axis=0)
        path_state = tf.concat([
            tf.expand_dims(f_['traffic'][0:f_["n_paths"]],axis=1),
            tf.zeros(shape)
        ], axis=1)

        links = f_['links']
        paths = f_['paths']
        seqs=  f_['sequences']

        for _ in range(self.hparams.T):

            h_tild = tf.gather(link_state,links)

            ids=tf.stack([paths, seqs], axis=1)
            max_len = tf.reduce_max(seqs)+1
            shape = tf.stack([f_['n_paths'], max_len, self.hparams.link_state_dim])
            lens = tf.math.segment_sum(data=tf.ones_like(paths),
                                    segment_ids=paths)

            link_inputs = tf.scatter_nd(ids, h_tild, shape)
            outputs, path_state = tf.nn.dynamic_rnn(self.path_update,
                                                    link_inputs,
                                                    sequence_length=lens,
                                                    initial_state = path_state,
                                                    dtype=tf.float32)
            m = tf.gather_nd(outputs,ids)
            m = tf.math.unsorted_segment_sum(m, links ,f_['n_links'])

            #Keras cell expects a list
            link_state,_ = self.edge_update(m, [link_state])

        if self.hparams.learn_embedding:
            r = self.readout(path_state,training=training)
        else:
            r = self.readout(tf.stop_gradient(path_state),training=training)

        return r

def model_fn(
   features, # This is batch_features from input_fn
   labels,   # This is batch_labrange
   mode,     # An instance of tf.estimator.ModeKeys
   params):  # Additional configuration


    model = ComnetModel(params)
    model.build()

    def fn(x):
        r = model(x,training=mode==tf.estimator.ModeKeys.TRAIN)
        return r

    predictions = fn(features)

    predictions = tf.squeeze(predictions)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode,
            predictions={'predictions':predictions})

    loss =  tf.losses.mean_squared_error(
        labels=labels,
        predictions = predictions,
        reduction=tf.losses.Reduction.MEAN
    )

    regularization_loss = sum(model.losses)
    total_loss = loss + regularization_loss

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('regularization_loss', regularization_loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss,
            eval_metric_ops={
                'label/mean': tf.metrics.mean(labels),
                'prediction/mean': tf.metrics.mean(predictions),
                'mae': tf.metrics.mean_absolute_error(labels, predictions),
                'rho': tf.contrib.metrics.streaming_pearson_correlation(labels=labels, predictions=predictions),
                'mre': tf.metrics.mean_relative_error(labels, predictions, labels)
            }
        )

    assert mode == tf.estimator.ModeKeys.TRAIN


    trainables = model.variables
    grads = tf.gradients(total_loss, trainables)
    grad_var_pairs = zip(grads, trainables)

    summaries = [tf.summary.histogram(var.op.name, var) for var in trainables]
    summaries += [tf.summary.histogram(g.op.name, g) for g in grads if g is not None]

    decayed_lr = tf.train.exponential_decay(params.learning_rate,
                                            tf.train.get_global_step(), 82000,
                                            0.8, staircase=True)
    optimizer = tf.train.AdamOptimizer(decayed_lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(grad_var_pairs,
            global_step=tf.train.get_global_step())

    logging_hook = tf.train.LoggingTensorHook(
        {"Training loss": loss}, every_n_iter=10)

    return tf.estimator.EstimatorSpec(mode,
                                      loss=loss,
                                      train_op=train_op,
                                      training_hooks=[logging_hook]
                                      )


hparams = tf.contrib.training.HParams(
    link_state_dim=4,
    path_state_dim=2,
    T=3,
    readout_units=8,
    learning_rate=0.001,
    batch_size=32,
    dropout_rate=0.5,
    l2=0.1,
    l2_2=0.01,
    learn_embedding=True # If false, only the readout is trained
)


def train(args):
    print(args)
    tf.logging.set_verbosity('INFO')

    if args.hparams:
        hparams.parse(args.hparams)

    my_checkpointing_config = tf.estimator.RunConfig(
        save_checkpoints_secs=10*60,  # Save checkpoints every 10 minutes
        keep_checkpoint_max=20  # Retain the 10 most recent checkpoints.
    )

    estimator = tf.estimator.Estimator(
        model_fn = model_fn,
        model_dir=args.model_dir,
        params=hparams,
        warm_start_from=args.warm,
        config=my_checkpointing_config
        )

    train_spec = tf.estimator.TrainSpec(input_fn=lambda:tfrecord_input_fn(args.train,hparams,shuffle_buf=args.shuffle_buf,target=args.target),max_steps=args.train_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:tfrecord_input_fn(args.eval_,hparams,shuffle_buf=None,target=args.target),throttle_secs=10*60)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

def extract_links(n, connections, link_cap):
    A = np.zeros((n,n))

    for a,c in zip(A,connections):
        a[c]=1

    G=nx.from_numpy_array(A, create_using=nx.DiGraph())
    edges=list(G.edges)
    capacities_links = []
    # The edges 0-2 or 2-0 can exist. They are duplicated (up and down) and they must have same capacity.
    for e in edges:
        if str(e[0])+':'+str(e[1]) in link_cap:
            capacity = link_cap[str(e[0])+':'+str(e[1])]
            capacities_links.append(capacity)
        elif str(e[1])+':'+str(e[0]) in link_cap:
            capacity = link_cap[str(e[1])+':'+str(e[0])]
            capacities_links.append(capacity)
        else:
            print("ERROR IN THE DATASET!")
            exit()
    return edges, capacities_links

def make_paths(R,connections, link_cap):
    n = R.shape[0]
    edges, capacities_links = extract_links(n, connections, link_cap)
    paths=[]
    for i in range(n):
        for j in range(n):
            if i != j:
                paths.append([edges.index(tup) for tup in pairwise(genPath(R,i,j,connections))])
    return paths, capacities_links

class NewParser:
    netSize = 0
    offsetDelay = 0
    hasPacketGen = True

    def __init__(self,netSize):
        self.netSize = netSize
        self.offsetDelay = netSize*netSize*3

    def getBwPtr(self,src,dst):
        return ((src*self.netSize + dst)*3)
    def getGenPcktPtr(self,src,dst):
        return ((src*self.netSize + dst)*3 + 1)
    def getDropPcktPtr(self,src,dst):
        return ((src*self.netSize + dst)*3 + 2)
    def getDelayPtr(self,src,dst):
        return (self.offsetDelay + (src*self.netSize + dst)*7)
    def getJitterPtr(self,src,dst):
        return (self.offsetDelay + (src*self.netSize + dst)*7 + 6)

def ned2lists(fname):
    channels = []
    link_cap = {}
    with open(fname) as f:
        p = re.compile(r'\s+node(\d+).port\[(\d+)\]\s+<-->\s+Channel(\d+)kbps+\s<-->\s+node(\d+).port\[(\d+)\]')
        for line in f:
            m=p.match(line)
            if m:
                auxList = []
                it = 0
                for elem in list(map(int,m.groups())):
                    if it!=2:
                        auxList.append(elem)
                    it = it + 1
                channels.append(auxList)
                link_cap[(m.groups()[0])+':'+str(m.groups()[3])] = int(m.groups()[2])

    n=max(map(max, channels))+1
    connections = [{} for i in range(n)]
    # Shape of connections[node][port] = node connected to
    for c in channels:
        connections[c[0]][c[1]]=c[2]
        connections[c[2]][c[3]]=c[0]
    # Connections store an array of nodes where each node position correspond to
    # another array of nodes that are connected to the current node
    connections = [[v for k,v in sorted(con.items())]
                   for con in connections ]
    return connections,n, link_cap

def get_corresponding_values(posParser, line, n, bws, delays, jitters):
    bws.fill(0)
    delays.fill(0)
    jitters.fill(0)
    it = 0
    for i in range(n):
        for j in range(n):
            if i!=j:
                delay = posParser.getDelayPtr(i, j)
                jitter = posParser.getJitterPtr(i, j)
                traffic = posParser.getBwPtr(i, j)
                bws[it] = float(line[traffic])
                delays[it] = float(line[delay])
                jitters[it] = float(line[jitter])
                it = it + 1

def make_tfrecord2(directory, tf_file, ned_file, routing_file, data_file):
    con,n,link_cap = ned2lists(ned_file)
    posParser = NewParser(n)

    R = load_routing(routing_file)
    paths, link_capacities = make_paths(R, con, link_cap)

    n_paths = len(paths)
    n_links = max(max(paths)) + 1
    a = np.zeros(n_paths)
    d = np.zeros(n_paths)
    j = np.zeros(n_paths)

    tfrecords_dir = directory+"tfrecords/"

    if not os.path.exists(tfrecords_dir):
        os.makedirs(tfrecords_dir)

    link_indices, path_indices, sequ_indices = make_indices(paths)
    n_total = len(path_indices)

    writer = tf.python_io.TFRecordWriter(tfrecords_dir + tf_file)

    for line in data_file:
        line = line.decode().split(',')
        get_corresponding_values(posParser, line, n, a, d, j)

        example = tf.train.Example(features=tf.train.Features(feature={
            'traffic':_float_features(a),
            'delay':_float_features(d),
            'jitter':_float_features(j),
            'link_capacity': _float_features(link_capacities),
            'links':_int64_features(link_indices),
            'paths':_int64_features(path_indices),
            'sequences':_int64_features(sequ_indices),
            'n_links':_int64_feature(n_links),
            'n_paths':_int64_feature(n_paths),
            'n_total':_int64_feature(n_total)
        }
        ))

        writer.write(example.SerializeToString())
    writer.close()

def data(args):
    directory = args.d[0]
    nodes_dir = directory.split('/')[-1]
    if (nodes_dir==''):
        nodes_dir=directory.split('/')[-2]

    ned_file = ""
    if nodes_dir=="geant2bw":
        ned_file = directory+"Network_geant2bw.ned"
    elif nodes_dir=="synth50bw":
        ned_file = directory+"Network_synth50bw.ned"
    elif nodes_dir=="nsfnetbw":
        ned_file = directory+"Network_nsfnetbw.ned"

    for filename in os.listdir(directory):
        if filename.endswith(".tar.gz"):
            print(filename)
            tf_file = filename.split('.')[0]+".tfrecords"
            tar = tarfile.open(directory+filename, "r:gz")

            dir_info = tar.next()
            if (not dir_info.isdir()):
                print("Tar file with wrong format")
                exit()

            delay_file = tar.extractfile(dir_info.name + "/simulationResults.txt")
            routing_file = tar.extractfile(dir_info.name + "/Routing.txt")

            tf.logging.info('Starting ', delay_file)
            make_tfrecord2(directory, tf_file,ned_file,routing_file,delay_file)

    directory_tfr = directory+"tfrecords/"

    tfr_train = directory_tfr+"train/"
    tfr_eval = directory_tfr+"evaluate/"
    if not os.path.exists(tfr_train):
        os.makedirs(tfr_train)

    if not os.path.exists(tfr_eval):
        os.makedirs(tfr_eval)

    tfrecords = glob.glob(directory_tfr+ '*.tfrecords')
    training = len(tfrecords) * 0.8
    train_samples = random.sample(tfrecords, int(training))
    evaluate_samples = list(set(tfrecords) - set(train_samples))

    for file in train_samples:
        file_name = file.split('/')[-1]
        os.rename(file, tfr_train + file_name)

    for file in evaluate_samples:
        file_name = file.split('/')[-1]
        os.rename(file, tfr_eval + file_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RouteNet: a Graph Neural Network model for computer network modeling')

    subparsers = parser.add_subparsers(help='sub-command help')
    parser_data = subparsers.add_parser('data', help='data processing')
    parser_data.add_argument('-d', help='data file', type=str, required=True, nargs='+')
    parser_data.set_defaults(func=data)

    parser_train = subparsers.add_parser('train', help='Train options')
    parser_train.add_argument('--hparams', type=str,help='Comma separated list of "name=value" pairs.')
    parser_train.add_argument('--train', help='Train Tfrecords files', type=str, nargs='+')
    parser_train.add_argument('--eval_', help='Evaluation Tfrecords files', type=str, nargs='+')
    parser_train.add_argument('--model_dir', help='Model directory', type=str)
    parser_train.add_argument('--train_steps', help='Training steps', type=int, default=100)
    parser_train.add_argument('--eval_steps', help='Evaluation steps, defaul None= all', type=int, default=None)
    parser_train.add_argument('--shuffle_buf', help="Buffer size for samples shuffling", type=int, default=10000)
    parser_train.add_argument('--target', help="Predicted variable", type=str, default='delay')
    parser_train.add_argument('--warm',help = "Warm start from", type=str, default=None)
    parser_train.set_defaults(func=train)
    parser_train.set_defaults(name="Train")

    args = parser.parse_args()
    args.func(args)