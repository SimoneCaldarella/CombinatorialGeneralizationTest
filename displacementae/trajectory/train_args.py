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
# @title          :displacementae/homomorphism/train_args.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :10/03/2022
# @version        :1.0
# @python_version :3.7.4

import argparse
from datetime import datetime

import utils.args as args

from grouprepr.representation_utils import Representation


def parse_cmd_arguments(representation=Representation.BLOCK_ROTS,
                        description='', argv=None):
    mode = 'trajectory'
    curr_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if not description:
        description = 'Rollout'
    dout_dir = './out/run_'+curr_date
    parser = argparse.ArgumentParser(description=description)
    args.data_args(parser, mode)
    args.train_args(parser)
    args.net_args(parser)
    args.misc_args(parser, dout_dir)
    args.group_repr_args(parser, representation)

    config = parser.parse_args(args=argv)

    config.intervene = True
    return config
