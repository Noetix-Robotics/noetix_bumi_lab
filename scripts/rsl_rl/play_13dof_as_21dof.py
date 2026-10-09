# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

"""Play a 13-DOF checkpoint and additionally export a 21-DOF deployment ONNX.

This entry point deliberately reuses the normal play implementation. The
standard ``play.py`` remains unchanged and still exports only ``policy.onnx``.
"""

import play as base_play

from rsl_rl.utils.exporter import export_policy_as_onnx_fixed_upper_body


_normal_export = base_play.export_policy_as_onnx
_normal_attach_metadata = base_play.attach_onnx_metadata
_last_export = {}


def _export_and_remember(policy, path, normalizer=None, filename="policy.onnx", verbose=False):
    _normal_export(policy, path=path, normalizer=normalizer, filename=filename, verbose=verbose)
    _last_export.update(policy=policy, path=path, normalizer=normalizer)


def _attach_metadata_and_export_adapter(env, path, filename="policy.onnx"):
    _normal_attach_metadata(env=env, path=path, filename=filename)
    if filename != "policy.onnx" or not _last_export:
        return

    adapter_filename = "policy_21dof_fixed_upper.onnx"
    exported = export_policy_as_onnx_fixed_upper_body(
        _last_export["policy"],
        env=env,
        path=_last_export["path"],
        normalizer=_last_export["normalizer"],
        filename=adapter_filename,
    )
    if not exported:
        raise RuntimeError("This play entry point requires a subset-DOF action policy.")
    _normal_attach_metadata(env=env, path=path, filename=adapter_filename)


base_play.export_policy_as_onnx = _export_and_remember
base_play.attach_onnx_metadata = _attach_metadata_and_export_adapter


if __name__ == "__main__":
    base_play.EXPORT_POLICY = True
    base_play.CONTROL_ROBOT = True
    base_play.main()
    base_play.simulation_app.close()
