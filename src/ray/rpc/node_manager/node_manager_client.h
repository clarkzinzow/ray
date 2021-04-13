// Copyright 2017 The Ray Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <grpcpp/grpcpp.h>

#include <thread>

#include "ray/common/status.h"
#include "ray/rpc/grpc_callback_client.h"
#include "ray/util/logging.h"
#include "src/ray/protobuf/node_manager.grpc.pb.h"
#include "src/ray/protobuf/node_manager.pb.h"

namespace ray {
namespace rpc {

/// Client used for communicating with a remote node manager server.
class NodeManagerClient {
 public:
  /// Constructor.
  ///
  /// \param[in] address Address of the node manager server.
  /// \param[in] port Port of the node manager server.
  /// \param[in] io_context: The asio event loop state.
  NodeManagerClient(const std::string &address,
                    const int port,
                    instrumented_io_context &io_context) {
    grpc_client_ = std::make_unique<GrpcCallbackClient<NodeManagerService>>(
        address, port, io_context);
  };

  /// Get current node stats.
  UNARY_CALLBACK_RPC_CLIENT_METHOD(NodeManagerService,
                                   GetNodeStats,
                                   grpc_client_,
                                   /*method_timeout_ms*/ -1, )

  void GetNodeStats(const ClientCallback<GetNodeStatsReply> &callback) {
    GetNodeStatsRequest request;
    GetNodeStats(request, callback);
  }

  std::shared_ptr<grpc::Channel> Channel() const { return grpc_client_->Channel(); }

 private:
  /// Stub for this service.
  std::unique_ptr<GrpcCallbackClient<NodeManagerService>> grpc_client_;
};

/// Client used by workers for communicating with a node manager server.
class NodeManagerWorkerClient
    : public std::enable_shared_from_this<NodeManagerWorkerClient> {
 public:
  /// Constructor.
  ///
  /// \param[in] address Address of the node manager server.
  /// \param[in] port Port of the node manager server.
  static std::shared_ptr<NodeManagerWorkerClient> make(
      const std::string &address, const int port, instrumented_io_context &io_context) {
    auto instance = new NodeManagerWorkerClient(address, port, io_context);
    return std::shared_ptr<NodeManagerWorkerClient>(instance);
  }

  std::shared_ptr<grpc::Channel> Channel() const { return grpc_client_->Channel(); }

  /// Update cluster resource usage.
  UNARY_CALLBACK_RPC_CLIENT_METHOD(NodeManagerService,
                                   UpdateResourceUsage,
                                   grpc_client_,
                                   /*method_timeout_ms*/ -1, )

  /// Request a resource report.
  UNARY_CALLBACK_RPC_CLIENT_METHOD(NodeManagerService,
                                   RequestResourceReport,
                                   grpc_client_,
                                   /*method_timeout_ms*/ -1, )

  /// Get a resource load
  UNARY_CALLBACK_RPC_CLIENT_METHOD(NodeManagerService,
                                   GetResourceLoad,
                                   grpc_client_,
                                   /*method_timeout_ms*/ -1, )

  /// Get a resource load
  UNARY_CALLBACK_RPC_CLIENT_METHOD(NodeManagerService,
                                   NotifyGCSRestart,
                                   grpc_client_,
                                   /*method_timeout_ms*/ -1, )

  /// Request a worker lease.
  UNARY_CALLBACK_RPC_CLIENT_METHOD(NodeManagerService,
                                   RequestWorkerLease,
                                   grpc_client_,
                                   /*method_timeout_ms*/ -1, )

  /// Return a worker lease.
  UNARY_CALLBACK_RPC_CLIENT_METHOD(NodeManagerService,
                                   ReturnWorker,
                                   grpc_client_,
                                   /*method_timeout_ms*/ -1, )

  /// Report task backlog information.
  UNARY_CALLBACK_RPC_CLIENT_METHOD(NodeManagerService,
                                   ReportWorkerBacklog,
                                   grpc_client_,
                                   /*method_timeout_ms*/ -1, )

  /// Release unused workers.
  UNARY_CALLBACK_RPC_CLIENT_METHOD(NodeManagerService,
                                   ReleaseUnusedWorkers,
                                   grpc_client_,
                                   /*method_timeout_ms*/ -1, )

  /// Shutdown the raylet gracefully
  UNARY_CALLBACK_RPC_CLIENT_METHOD(NodeManagerService,
                                   ShutdownRaylet,
                                   grpc_client_,
                                   /*method_timeout_ms*/ -1, )

  /// Cancel a pending worker lease request.
  UNARY_CALLBACK_RPC_CLIENT_METHOD(NodeManagerService,
                                   CancelWorkerLease,
                                   grpc_client_,
                                   /*method_timeout_ms*/ -1, )

  /// Request prepare resources for an atomic placement group creation.
  UNARY_CALLBACK_RPC_CLIENT_METHOD(NodeManagerService,
                                   PrepareBundleResources,
                                   grpc_client_,
                                   /*method_timeout_ms*/ -1, )

  /// Request commit resources for an atomic placement group creation.
  UNARY_CALLBACK_RPC_CLIENT_METHOD(NodeManagerService,
                                   CommitBundleResources,
                                   grpc_client_,
                                   /*method_timeout_ms*/ -1, )

  /// Return resource lease.
  UNARY_CALLBACK_RPC_CLIENT_METHOD(NodeManagerService,
                                   CancelResourceReserve,
                                   grpc_client_,
                                   /*method_timeout_ms*/ -1, )

  /// Notify the raylet to pin the provided object IDs.
  UNARY_CALLBACK_RPC_CLIENT_METHOD(NodeManagerService,
                                   PinObjectIDs,
                                   grpc_client_,
                                   /*method_timeout_ms*/ -1, )

  /// Trigger global GC across the cluster.
  UNARY_CALLBACK_RPC_CLIENT_METHOD(NodeManagerService,
                                   GlobalGC,
                                   grpc_client_,
                                   /*method_timeout_ms*/ -1, )

  /// Ask the raylet to spill an object to external storage.
  UNARY_CALLBACK_RPC_CLIENT_METHOD(NodeManagerService,
                                   RequestObjectSpillage,
                                   grpc_client_,
                                   /*method_timeout_ms*/ -1, )

  /// Release unused bundles.
  UNARY_CALLBACK_RPC_CLIENT_METHOD(NodeManagerService,
                                   ReleaseUnusedBundles,
                                   grpc_client_,
                                   /*method_timeout_ms*/ -1, )

  /// Get the system config from Raylet.
  UNARY_CALLBACK_RPC_CLIENT_METHOD(NodeManagerService,
                                   GetSystemConfig,
                                   grpc_client_,
                                   /*method_timeout_ms*/ -1, )

  /// Get all the task information from the node.
  UNARY_CALLBACK_RPC_CLIENT_METHOD(NodeManagerService,
                                   GetTasksInfo,
                                   grpc_client_,
                                   /*method_timeout_ms*/ -1, )

  /// Get all the object information from the node.
  UNARY_CALLBACK_RPC_CLIENT_METHOD(NodeManagerService,
                                   GetObjectsInfo,
                                   grpc_client_,
                                   /*method_timeout_ms*/ -1, )

  UNARY_CALLBACK_RPC_CLIENT_METHOD(NodeManagerService,
                                   GetTaskFailureCause,
                                   grpc_client_,
                                   /*method_timeout_ms*/ -1, )

 private:
  /// Constructor.
  ///
  /// \param[in] address Address of the node manager server.
  /// \param[in] port Port of the node manager server.
  /// \param[in] io_context: The asio event loop state.
  NodeManagerWorkerClient(const std::string &address,
                          const int port,
                          instrumented_io_context &io_context) {
    grpc_client_ = std::make_unique<GrpcCallbackClient<NodeManagerService>>(
        address, port, io_context);
  };

  /// The gRPC client.
  std::unique_ptr<GrpcCallbackClient<NodeManagerService>> grpc_client_;
};

}  // namespace rpc
}  // namespace ray
