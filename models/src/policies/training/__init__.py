"""
Training utilities and pipeline components.

Key exports:
  - RobotProcessor / ProcessorConfig  (data normalization)
  - RobotFlowPolicyWrapper            (model + processor bundle, also inference entrypoint)
  - RobotFlowTrainer                  (training loop)
  - RobotFlowEvaluator                (validation)
  - ExperimentLogger                  (W&B logging)
  - StatusMonitor                     (rolling metrics)
  - CUDAPrefetcher / move_to_device   (device utilities)
"""
