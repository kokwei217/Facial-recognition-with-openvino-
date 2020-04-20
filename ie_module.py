"""
 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for thbe specific language governing permissions and
 limitations under the License.
"""

import logging as log
import os.path as osp
from itertools import cycle
from collections import deque

from openvino.inference_engine import IECore, IENetwork

ie = IECore()

class InferenceContext:
    def deploy_model(self, model, device, max_requests=1):
        deployed_model = ie.load_network(network=model, device_name= device, num_requests = max_requests)
        return deployed_model


class Module(object):
    def __init__(self, model):
        self.model = model
        self.device_model = None
        self.max_requests = 0
        self.active_requests = 0

        self.enable_multi = False
        self.available_NCS = []
        self.device_models = []
        self.clear()

    def deploy(self, device, context, queue_size=1, enable_multi = False):
        self.available_NCS = ie.get_metric("MYRIAD" , "AVAILABLE_DEVICES")
        self.enable_multi = enable_multi
        self.context = context
        self.max_requests = queue_size
        if(self.enable_multi and len(self.available_NCS)>1):
            log.info("Multi Stick Approach")
            for ncs_id in self.available_NCS:
                ncs = device +"." + ncs_id
                log.info("Loading on '%s'" %ncs)
                self.device_models.append(context.deploy_model(self.model, ncs, self.max_requests))
        else:
            log.info("Single Stick Approach, Loading on %s" %device)       
            self.device_model = context.deploy_model(
                self.model, device, self.max_requests)

        self.model = None

    def enqueue(self, input):
        # for face detection, active request is only 1 which is the one frame

        self.clear()

        if self.max_requests <= self.active_requests:
            log.warning("Processing request rejected - too many requests")
            return False
        if(self.enable_multi and len(self.available_NCS)>1):
            # print(self.active_requests)
            if(self.active_requests%2 == 0):
                self.device_models[0].start_async(self.active_requests,input)
            else:
                self.device_models[1].start_async(self.active_requests,input)
        else:
            self.device_model.start_async(self.active_requests, input)

        self.active_requests += 1
        return True

    def wait(self):
        if self.active_requests <= 0:
            return

        self.perf_stats = [None, ] * self.active_requests
        self.outputs = [None, ] * self.active_requests
        # i is the request id of each active inference requests
        for i in range(self.active_requests):
            # wait until the result is ready
            if(self.enable_multi and len(self.available_NCS)>1):
                if(self.active_requests%2==1):
                    self.device_models[0].requests[i].wait()
                    self.outputs[i] = self.device_models[0].requests[i].outputs
                else:
                    self.device_models[1].requests[i].wait()
                    self.outputs[i] = self.device_models[1].requests[i].outputs
            else:
                self.device_model.requests[i].wait()
                self.outputs[i] = self.device_model.requests[i].outputs
            # self.perf_stats[i] = self.device_model.requests[i].get_perf_counts()
        self.active_requests = 0

    def get_outputs(self):
        self.wait()
        return self.outputs

    def get_performance_stats(self):
        return self.perf_stats

    def clear(self):
        self.perf_stats = []
        self.outputs = []
