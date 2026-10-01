# Third-party notices

## Locomotion reference implementation

The locomotion reproduction uses the external model, metadata, training
configuration and evaluator from
[yezzzzye/g1_walk_isaaclab_mujoco](https://github.com/yezzzzye/g1_walk_isaaclab_mujoco).
Those dependencies and checkpoints remain in a separate local checkout.
The published Sim2Sim video renders that external G1 model under a locally
trained policy; it is a simulation result, not an original robot asset or
an endorsement by Unitree. See the reproduction result summary for revision
and policy provenance. No third-party source or checkpoint is bundled in
the new result package.

The MuJoCo recordings and standing experiments in this repository use the
Unitree G1 model and scene from
[unitreerobotics/unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco).
The robot model is an external dependency; this repository does not claim
ownership of it or include its model files. The locally used Unitree checkout
provided the following notice. The Unitree name is used for identification,
not endorsement.

## Unitree MuJoCo — BSD 3-Clause License

Copyright (c) 2016-2024 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
