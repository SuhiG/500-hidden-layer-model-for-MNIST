from __future__ import print_function
import numpy as np 
import theano as t 
import theano.tensor as tt
import pickle, gzip
import timeit

print("Using device", t.config.device)

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f,encoding="latin1")#load the mnist dataset (use 'latin1' for python3)

#print("Loading data")
#with gzip.open(data_dir + "mnist.pkl.gz", 'rb') as f:
   # train_set, valid_set, test_set = pickle.load(f)

train_set_x = t.shared(np.asarray(train_set[0],  dtype=t.config.floatX))
train_set_y = t.shared(np.asarray(train_set[1],  dtype='int32'))

print("Building model")

batch_size = 600 #takes the first 600 samples for training
n_in=28 * 28#inputs
n_hidden=500#hidden layers
n_out=10#outputs

x = tt.matrix('x') #input matrix(because inputs are pictures)
y = tt.ivector('y')#output vector(because outputs are single numbers)

def shared_zeros(shape, dtype=t.config.floatX, name='', n=None):
    shape = shape if n is None else (n,) + shape
    return t.shared(np.zeros(shape, dtype=dtype), name=name)

def shared_glorot_uniform(shape, dtype=t.config.floatX, name='', n=None):
    if isinstance(shape, int):
        high = np.sqrt(6. / shape)
    else:
        high = np.sqrt(6. / (np.sum(shape[:2]) * np.prod(shape[2:])))
    shape = shape if n is None else (n,) + shape
    return t.shared(np.asarray(
        np.random.uniform(
            low=-high,
            high=high,
            size=shape),
        dtype=dtype), name=name)

#below, the reason that we use shared is to use those variables through out the run
W1 = shared_glorot_uniform( (n_in, n_hidden), name='W1' )
b1 = shared_zeros( (n_hidden), name='b1' )#hte reason that we don't use tt.zeros() because we have to give the size of the matrix 

hidden_output = tt.tanh(tt.dot(x, W1) + b1)#using tanh as the activation function for the hidden layers

W2 = shared_zeros( (n_hidden, n_out), name='W2' )
b2 = shared_zeros( (n_out,), name='b2' )


params = [W1,b1,W2,b2]

model = tt.nnet.softmax(tt.dot(hidden_output, W2) + b2)

y_pred = tt.argmax(model, axis=1)#selecting the largest number from the output matrix
error = tt.mean(tt.neq(y_pred, y))#calculating the error by 

cost = -tt.mean(tt.log(model)[tt.arange(y.shape[0]), y]) + 0.0001 * (W1 ** 2).sum() + 0.0001 * (W2 ** 2).sum()

g_params = tt.grad(cost=cost, wrt=params)

learning_rate=0.01
updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(params, g_params)
    ]

index = tt.lscalar()

train_model = t.function(
    inputs=[index],
    outputs=[cost,error],
    updates=updates,
    givens={
        x: train_set_x[index * batch_size: (index + 1) * batch_size],
        y: train_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

validate_model = t.function(
    inputs=[x,y],
    outputs=[cost,error]
)


print("Training")

n_epochs = 100
n_train_batches = train_set[0].shape[0] // batch_size

n_iters = n_epochs * n_train_batches
train_loss = np.zeros(n_iters)
train_error = np.zeros(n_iters)

validation_interval = 100
n_valid_batches = valid_set[0].shape[0] // batch_size
valid_loss = np.zeros(n_iters //validation_interval)
valid_error = np.zeros(n_iters // validation_interval)


start_time = timeit.default_timer()
for epoch in range(n_epochs):
    for minibatch_index in range(n_train_batches):
        iteration = minibatch_index + n_train_batches * epoch
        train_loss[iteration], train_error[iteration] = train_model(minibatch_index)

        if iteration % validation_interval == 0 :
            val_iteration = iteration // validation_interval
            valid_loss[val_iteration], valid_error[val_iteration] = np.mean([
                    validate_model(
                        valid_set[0][i * batch_size: (i + 1) * batch_size],
                        np.asarray(valid_set[1][i * batch_size: (i + 1) * batch_size], dtype="int32")
                        )
                        for i in range(n_valid_batches)
                     ],axis=0)

            print('epoch {}, minibatch {}/{}, validation error {:02.2f} %, validation loss {}'.format(
                epoch,
                minibatch_index + 1,
                n_train_batches,
                valid_error[val_iteration] * 100,
                valid_loss[val_iteration]
            ))

end_time = timeit.default_timer()
print( end_time -start_time )