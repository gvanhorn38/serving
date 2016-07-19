# Classification Services

## Inputs and Outputs
The classification services expect the input to be a preprocessed image that has been flattened. "Prepocessed" in this case means that the image data is ready to be sent into the inference part of the model (i.e. already centered, normalized, and resized to the correct size, etc.). The output from the classification service is a sorted array of class scores along with their indices. 

## Basic Classifier Server
The basic classification server simply creates a [SessionBundle](../session_bundle/session_bundle.h) that is accessed to run the model. This server is passed a specific model to run, and does not check for new models. To change models, the server must be stopped and restarted. If the `--use_batching` command line switch is used, then the SessionBundle is wrapped by a [BatchingSession](../batching/batching_session.h) object, which uses a [BasicBatchSceduler](../batching/basic_batch_scheduler) to do the batching. Command line arguments can be passed in to configure the batching. See the [BatchingSession](../batching/batching_session.h) and [BasicBatchSceduler](../batching/basic_batch_scheduler) header files for more details on these parameters:

| Command Line Flag | Default | Description |
|-------------------|---------|-------------|
| thread_pool_name     | "batch_threads" | The name to use for the pool of batch threads. |
| num_batch_threads    | 1 | The number of threads to use to process batches. Must be >= 1, and should be tuned carefully.|
| max_batch_size       | 1000 | The scheduler may form batches of any size between 1 and this number (inclusive). |
| batch_timeout_micros |1000 | If a task has been enqueued for this amount of time (in microseconds), and a thread is available, the scheduler will immediately form a batch from enqueued tasks and assign the batch to the thread for processing, even if the batch's size is below `--max_batch_size`.|
| max_enqueued_batches | 1 | The maximum allowable number of enqueued (accepted by Schedule() but not yet being processed on a batch thread) tasks in terms of batches. If this limit is reached, Schedule() will return an UNAVAILABLE error.|
| allowed_batch_sizes  | "" | When the batch scheduler forms a batch of size N, the batch size is rounded up to the smallest value M in 'allowed_batch_sizes' s.t. M >= N. The tensors submitted to the underlying Session are padded with M-N repeats of one of the first N entries (i.e. a guaranteed valid entry). The last M-N entries of the output tensors are ignored. If left empty, no rounding/padding is performed.|

Build the server:
```shell
$ bazel build tensorflow_serving/extra_examples/basic_classifier_server
``` 

Run the server:
```shell
$ bazel-bin/tensorflow_serving/extra_examples/basic_classifier_server --port=9000 \
--use_batching \
--thread_pool_name="inference_server_batch_threads" \
--num_batch_threads=1 \
--max_batch_size=64 \
--batch_timeout_micros=1000 \
--max_enqueued_batches=1000 \
--allowed_batch_sizes="1 8 16 32 64" \
/tmp/model/00000003/
```

## Classification Client
The classification client is a python script that is passed the url of a classification server and a path to a JPEG image. The image is loaded, preprocessed and sent to the server. The results and the time for classification are printed out.
```shell
$ python tensorflow_serving/extra_examples/classifier_client.py --server=localhost:9000 \
--image=/tmp/test.jpg
```

# Modifying the Protcol Buffer
If you make modifications to the protocol buffers defining the inputs, outputs, or services then you need to recompile the file. You'll need to install the grpcio-tools python package:
```shell
$ pip install grpcio-tools
```

You can then compile the file:
```shell
$ python -m grpc.tools.protoc -Itensorflow_serving/extra_examples \
--python_out=tensorflow_serving/extra_examples/ \
--grpc_python_out=tensorflow_serving/extra_examples/ \
tensorflow_serving/extra_examples/services.proto 
```