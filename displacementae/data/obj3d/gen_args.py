#!/usr/bin/env python3
# Copyright 2021 Hamza Keurti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @title          :displacementae/data/obj3d/gen_args.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :01/09/2022
# @version        :1.0
# @python_version :3.7.4

import argparse

import utils.args as args

def parse_cmd_arguments():
    parser = argparse.ArgumentParser()
    args.data_gen_args(parser)
    config =parser.parse_args()
    return config

