/*
This is a basic server implementation for a classification service. This code will start
a grpc server with the option of adding batching support.

This code is adapted from the mnist_inference.cc example file.

*/

/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <chrono>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "grpc++/security/server_credentials.h"
#include "grpc++/server.h"
#include "grpc++/server_builder.h"
#include "grpc++/server_context.h"
#include "grpc++/support/status.h"
#include "grpc++/support/status_code_enum.h"
#include "grpc/grpc.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow_serving/extra_examples/services.grpc.pb.h"
#include "tensorflow_serving/extra_examples/services.pb.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_factory.h"
#include "tensorflow_serving/session_bundle/manifest.pb.h"
#include "tensorflow_serving/session_bundle/session_bundle.h"
#include "tensorflow_serving/session_bundle/signature.h"

using std::string;
using std::stringstream;
using grpc::InsecureServerCredentials;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using grpc::StatusCode;
using tensorflow::serving::ClassificationSignature;
using tensorflow::serving::ClassificationRequest;
using tensorflow::serving::ClassificationResponse;
using tensorflow::serving::ClassificationService;
using tensorflow::serving::BatchingParameters;
using tensorflow::serving::SessionBundle;
using tensorflow::serving::SessionBundleConfig;
using tensorflow::serving::SessionBundleFactory;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::TensorShape;

namespace {

const int kImageSize = 299;
const int kNumChannels = 3;
const int kImageDataSize = kImageSize * kImageSize * kNumChannels;

// Creates a gRPC Status from a TensorFlow Status.
Status ToGRPCStatus(const tensorflow::Status& status) {
  return Status(static_cast<grpc::StatusCode>(status.code()),
                status.error_message());
}

class ClassificationServiceImpl final : public ClassificationService::Service {
 public:
  explicit ClassificationServiceImpl(std::unique_ptr<SessionBundle> bundle)
      : bundle_(std::move(bundle)) {
    signature_status_ = tensorflow::serving::GetClassificationSignature(
        bundle_->meta_graph_def, &signature_);
  }

  Status Classify(ServerContext* context, const ClassificationRequest* request,
                  ClassificationResponse* response) override {

    // Copy the image data into the input protocol buffer
    int batch_size = 1;
    Tensor input(tensorflow::DT_FLOAT, {batch_size, kImageDataSize});
    auto dst = input.flat_outer_dims<float>().data();
    std::copy_n(
        request->image_data().begin(),
        kImageDataSize, dst);

    std::vector<Tensor> outputs;
    
    // Run inference.
    if (!signature_status_.ok()) {
     return ToGRPCStatus(signature_status_);
    }
    
    auto start = std::chrono::high_resolution_clock::now();

    const tensorflow::Status status = bundle_->session->Run(
        {{signature_.input().tensor_name(), input}},
        {signature_.scores().tensor_name(), signature_.classes().tensor_name()}, {}, &outputs);
    if (!status.ok()) {
      return ToGRPCStatus(status);
    }
    
    auto finish = std::chrono::high_resolution_clock::now();

    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(finish-start);
    std::cout << "classification time: " << milliseconds.count() << "ms\n";

    // Copy the scores to the output protocol buffer
    const auto score_flat = outputs[0].flat<float>();
    auto scores = response->mutable_scores();
    for (int i = 0; i < score_flat.size(); ++i) {
      scores->Add(score_flat(i));
    }

    // Copy the classes to the output protocol buffer
    auto classes = response->mutable_classes();
    const auto class_flat = outputs[1].flat<int>();
    for (int i = 0; i < class_flat.size(); ++i) {
      classes->Add(class_flat(i));
    }

    return Status::OK;
  }

 private:
  std::unique_ptr<SessionBundle> bundle_;
  tensorflow::Status signature_status_;
  ClassificationSignature signature_;
};

void RunServer(int port, std::unique_ptr<SessionBundle> bundle) {
  // "0.0.0.0" is the way to listen on localhost in gRPC.
  const string server_address = "0.0.0.0:" + std::to_string(port);
  ClassificationServiceImpl service(std::move(bundle));
  ServerBuilder builder;
  std::shared_ptr<grpc::ServerCredentials> creds = InsecureServerCredentials();
  builder.AddListeningPort(server_address, creds);
  builder.RegisterService(&service);
  std::unique_ptr<Server> server(builder.BuildAndStart());
  LOG(INFO) << "Running...";
  server->Wait();
}

}  // namespace

// Usage String
const string usage_str = R"(Usage: inference --port=9000 \
  --use_batching \
  --thread_pool_name="inference_server_batch_threads" \
  --num_batch_threads=1 \
  --max_batch_size=64 \
  --batch_timeout_micros=1000 \
  --max_enqueued_batches=1000 \
  --allowed_batch_sizes="1 8 16 32 64" \
  /path/to/export)";

int main(int argc, char** argv) {
  
  // Command Line Arguments
  tensorflow::int32 port = 0;
  bool use_batching = false;
  string thread_pool_name = "batch_threads";
  tensorflow::int32 num_batch_threads = 1;
  tensorflow::int32 max_batch_size = 1000;
  tensorflow::int32 batch_timeout_micros = 1 * 1000; // 1 millisecond
  tensorflow::int32 max_enqueued_batches = 1000;
  string allowed_batch_sizes="";
  
  const bool parse_result =
      tensorflow::ParseFlags(&argc, argv, {
        tensorflow::Flag("port", &port), 
        tensorflow::Flag("use_batching", &use_batching),
        tensorflow::Flag("thread_pool_name", &thread_pool_name),
        tensorflow::Flag("num_batch_threads", &num_batch_threads),
        tensorflow::Flag("max_batch_size", &max_batch_size),
        tensorflow::Flag("batch_timeout_micros", &batch_timeout_micros),
        tensorflow::Flag("max_enqueued_batches", &max_enqueued_batches),
        tensorflow::Flag("allowed_batch_sizes", &allowed_batch_sizes)
      });
  if (!parse_result) {
    LOG(FATAL) << "Error parsing command line flags.";
  }

  if (argc != 2) {
    LOG(FATAL) << usage_str;
  }
  
  const string bundle_path(argv[1]);

  tensorflow::port::InitMain(argv[0], &argc, &argv);

  SessionBundleConfig session_bundle_config;
  if(use_batching){
    // Request batching 
    BatchingParameters* batching_parameters = session_bundle_config.mutable_batching_parameters();
    batching_parameters->mutable_thread_pool_name()->set_value(thread_pool_name);
    batching_parameters->mutable_num_batch_threads()->set_value(num_batch_threads);
    batching_parameters->mutable_max_batch_size()->set_value(max_batch_size);
    batching_parameters->mutable_batch_timeout_micros()->set_value(batch_timeout_micros); 
    batching_parameters->mutable_max_enqueued_batches()->set_value(max_enqueued_batches);
    if(allowed_batch_sizes.length() != 0){
      int batch_size;
      stringstream stream(allowed_batch_sizes);
      while(stream >> batch_size){
          batching_parameters->add_allowed_batch_sizes(batch_size);
      }
    }
  }
  // Load an exported TensorFlow model at the given path and create a SessionBundle object
  // for running inference with the model.
  std::unique_ptr<SessionBundleFactory> bundle_factory;
  TF_QCHECK_OK(
      SessionBundleFactory::Create(session_bundle_config, &bundle_factory));
  std::unique_ptr<SessionBundle> bundle(new SessionBundle);
  TF_QCHECK_OK(bundle_factory->CreateSessionBundle(bundle_path, &bundle));

  RunServer(port, std::move(bundle));

  return 0;
}
