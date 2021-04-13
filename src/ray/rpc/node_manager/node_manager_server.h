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

#include "ray/common/asio/instrumented_io_context.h"
#include "ray/rpc/grpc_callback_server.h"
#include "src/ray/protobuf/node_manager.grpc.pb.h"
#include "src/ray/protobuf/node_manager.pb.h"

namespace ray {
namespace rpc {

/// Interface of the `NodeManagerService`, see `src/ray/protobuf/node_manager.proto`.
class NodeManagerServiceHandler {
 public:
  /// Handlers. For all of the following handlers, the implementations can
  /// handle the request asynchronously. When handling is done, the
  /// `send_reply_callback` should be called. See
  /// src/ray/rpc/node_manager/node_manager_client.h and
  /// src/ray/protobuf/node_manager.proto for a description of the
  /// functionality of each handler.
  ///
  /// \param[in] request The request message.
  /// \param[out] reply The reply message.
  /// \param[in] send_reply_callback The callback to be called when the request is done.

  virtual void HandleUpdateResourceUsage(rpc::UpdateResourceUsageRequest request,
                                         rpc::UpdateResourceUsageReply *reply,
                                         rpc::SendReplyCallback send_reply_callback) = 0;

  virtual void HandleRequestResourceReport(
      rpc::RequestResourceReportRequest request,
      rpc::RequestResourceReportReply *reply,
      rpc::SendReplyCallback send_reply_callback) = 0;

  virtual void HandleGetResourceLoad(rpc::GetResourceLoadRequest request,
                                     rpc::GetResourceLoadReply *reply,
                                     rpc::SendReplyCallback send_reply_callback) = 0;

  virtual void HandleNotifyGCSRestart(rpc::NotifyGCSRestartRequest request,
                                      rpc::NotifyGCSRestartReply *reply,
                                      rpc::SendReplyCallback send_reply_callback) = 0;

  virtual void HandleRequestWorkerLease(RequestWorkerLeaseRequest request,
                                        RequestWorkerLeaseReply *reply,
                                        SendReplyCallback send_reply_callback) = 0;

  virtual void HandleReportWorkerBacklog(ReportWorkerBacklogRequest request,
                                         ReportWorkerBacklogReply *reply,
                                         SendReplyCallback send_reply_callback) = 0;

  virtual void HandleReturnWorker(ReturnWorkerRequest request,
                                  ReturnWorkerReply *reply,
                                  SendReplyCallback send_reply_callback) = 0;

  virtual void HandleReleaseUnusedWorkers(ReleaseUnusedWorkersRequest request,
                                          ReleaseUnusedWorkersReply *reply,
                                          SendReplyCallback send_reply_callback) = 0;

  virtual void HandleShutdownRaylet(ShutdownRayletRequest request,
                                    ShutdownRayletReply *reply,
                                    SendReplyCallback send_reply_callback) = 0;

  virtual void HandleCancelWorkerLease(rpc::CancelWorkerLeaseRequest request,
                                       rpc::CancelWorkerLeaseReply *reply,
                                       rpc::SendReplyCallback send_reply_callback) = 0;

  virtual void HandlePrepareBundleResources(
      rpc::PrepareBundleResourcesRequest request,
      rpc::PrepareBundleResourcesReply *reply,
      rpc::SendReplyCallback send_reply_callback) = 0;

  virtual void HandleCommitBundleResources(
      rpc::CommitBundleResourcesRequest request,
      rpc::CommitBundleResourcesReply *reply,
      rpc::SendReplyCallback send_reply_callback) = 0;

  virtual void HandleCancelResourceReserve(
      rpc::CancelResourceReserveRequest request,
      rpc::CancelResourceReserveReply *reply,
      rpc::SendReplyCallback send_reply_callback) = 0;

  virtual void HandlePinObjectIDs(PinObjectIDsRequest request,
                                  PinObjectIDsReply *reply,
                                  SendReplyCallback send_reply_callback) = 0;

  virtual void HandleGetNodeStats(GetNodeStatsRequest request,
                                  GetNodeStatsReply *reply,
                                  SendReplyCallback send_reply_callback) = 0;

  virtual void HandleGlobalGC(GlobalGCRequest request,
                              GlobalGCReply *reply,
                              SendReplyCallback send_reply_callback) = 0;

  virtual void HandleFormatGlobalMemoryInfo(FormatGlobalMemoryInfoRequest request,
                                            FormatGlobalMemoryInfoReply *reply,
                                            SendReplyCallback send_reply_callback) = 0;

  virtual void HandleRequestObjectSpillage(RequestObjectSpillageRequest request,
                                           RequestObjectSpillageReply *reply,
                                           SendReplyCallback send_reply_callback) = 0;

  virtual void HandleReleaseUnusedBundles(ReleaseUnusedBundlesRequest request,
                                          ReleaseUnusedBundlesReply *reply,
                                          SendReplyCallback send_reply_callback) = 0;

  virtual void HandleGetSystemConfig(GetSystemConfigRequest request,
                                     GetSystemConfigReply *reply,
                                     SendReplyCallback send_reply_callback) = 0;

  virtual void HandleGetTasksInfo(GetTasksInfoRequest request,
                                  GetTasksInfoReply *reply,
                                  SendReplyCallback send_reply_callback) = 0;

  virtual void HandleGetObjectsInfo(GetObjectsInfoRequest request,
                                    GetObjectsInfoReply *reply,
                                    SendReplyCallback send_reply_callback) = 0;

  virtual void HandleGetTaskFailureCause(GetTaskFailureCauseRequest request,
                                         GetTaskFailureCauseReply *reply,
                                         SendReplyCallback send_reply_callback) = 0;
};

/// NOTE: See src/ray/core_worker/core_worker.h on how to add a new grpc handler.
#define RAY_NODE_MANAGER_RPC_HANDLERS                                                \
  UNARY_CALLBACK_RPC_SERVICE_HANDLER(NodeManagerService, UpdateResourceUsage, -1)    \
  UNARY_CALLBACK_RPC_SERVICE_HANDLER(NodeManagerService, RequestResourceReport, -1)  \
  UNARY_CALLBACK_RPC_SERVICE_HANDLER(NodeManagerService, GetResourceLoad, -1)        \
  UNARY_CALLBACK_RPC_SERVICE_HANDLER(NodeManagerService, NotifyGCSRestart, -1)       \
  UNARY_CALLBACK_RPC_SERVICE_HANDLER(NodeManagerService, RequestWorkerLease, -1)     \
  UNARY_CALLBACK_RPC_SERVICE_HANDLER(NodeManagerService, ReportWorkerBacklog, -1)    \
  UNARY_CALLBACK_RPC_SERVICE_HANDLER(NodeManagerService, ReturnWorker, -1)           \
  UNARY_CALLBACK_RPC_SERVICE_HANDLER(NodeManagerService, ReleaseUnusedWorkers, -1)   \
  UNARY_CALLBACK_RPC_SERVICE_HANDLER(NodeManagerService, CancelWorkerLease, -1)      \
  UNARY_CALLBACK_RPC_SERVICE_HANDLER(NodeManagerService, PinObjectIDs, -1)           \
  UNARY_CALLBACK_RPC_SERVICE_HANDLER(NodeManagerService, GetNodeStats, -1)           \
  UNARY_CALLBACK_RPC_SERVICE_HANDLER(NodeManagerService, GlobalGC, -1)               \
  UNARY_CALLBACK_RPC_SERVICE_HANDLER(NodeManagerService, FormatGlobalMemoryInfo, -1) \
  UNARY_CALLBACK_RPC_SERVICE_HANDLER(NodeManagerService, PrepareBundleResources, -1) \
  UNARY_CALLBACK_RPC_SERVICE_HANDLER(NodeManagerService, CommitBundleResources, -1)  \
  UNARY_CALLBACK_RPC_SERVICE_HANDLER(NodeManagerService, CancelResourceReserve, -1)  \
  UNARY_CALLBACK_RPC_SERVICE_HANDLER(NodeManagerService, RequestObjectSpillage, -1)  \
  UNARY_CALLBACK_RPC_SERVICE_HANDLER(NodeManagerService, ReleaseUnusedBundles, -1)   \
  UNARY_CALLBACK_RPC_SERVICE_HANDLER(NodeManagerService, GetSystemConfig, -1)        \
  UNARY_CALLBACK_RPC_SERVICE_HANDLER(NodeManagerService, ShutdownRaylet, -1)         \
  UNARY_CALLBACK_RPC_SERVICE_HANDLER(NodeManagerService, GetTasksInfo, -1)           \
  UNARY_CALLBACK_RPC_SERVICE_HANDLER(NodeManagerService, GetObjectsInfo, -1)         \
  UNARY_CALLBACK_RPC_SERVICE_HANDLER(NodeManagerService, GetTaskFailureCause, -1)

CALLBACK_SERVICE(NodeManagerService, RAY_NODE_MANAGER_RPC_HANDLERS)

}  // namespace rpc
}  // namespace ray
