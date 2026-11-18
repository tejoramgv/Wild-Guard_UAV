"""Safety envelope and command arbitration tests."""

import math

from forest_bringup.safety import (
    CommandArbiter, FlightLimits, SafetySupervisor, VelocityCommand,
    clamp_command, rate_limit)


def test_clamp_horizontal_and_vertical():
    limits = FlightLimits(max_speed=2.0, max_vertical_speed=1.5, max_yaw_rate=1.0)
    out = clamp_command(VelocityCommand(3.0, 4.0, 9.0, 9.0), limits)
    assert math.isclose(math.hypot(out.vx, out.vy), 2.0)
    assert out.vz == 1.5 and out.wz == 1.0


def test_clamp_altitude():
    limits = FlightLimits(max_altitude=10.0, min_altitude=0.5)
    top = clamp_command(VelocityCommand(vz=1.0), limits, position_enu=(0, 0, 10.0))
    assert top.vz == 0.0
    bottom = clamp_command(VelocityCommand(vz=-1.0), limits, position_enu=(0, 0, 0.5))
    assert bottom.vz == 0.0
    mid = clamp_command(VelocityCommand(vz=1.0), limits, position_enu=(0, 0, 5.0))
    assert mid.vz == 1.0


def test_rate_limit():
    current = VelocityCommand()
    target = VelocityCommand(2.0, 0, 0, 0)
    out = rate_limit(current, target, max_acceleration=1.0, dt=0.2)
    assert math.isclose(out.vx, 0.2)
    out2 = rate_limit(out, target, max_acceleration=1.0, dt=10.0)
    assert out2.vx == 2.0


def test_arbiter_priority_and_timeout():
    arb = CommandArbiter(command_timeout=0.5)
    arb.set_manual(VelocityCommand(1.0), now=0.0)
    cmd, source = arb.select(0.0)
    assert source == "manual" and cmd.vx == 1.0

    arb.set_avoidance(VelocityCommand(0.0, 0.0, 0.0, 1.0), active=True, now=0.1)
    cmd, source = arb.select(0.1)
    assert source == "obstacle_avoidance" and cmd.wz == 1.0

    # No fresh manual command and avoidance not active -> hover.
    arb2 = CommandArbiter(command_timeout=0.5)
    arb2.set_manual(VelocityCommand(1.0), now=0.0)
    cmd, source = arb2.select(1.0)
    assert source == "command_timeout_hover" and cmd.is_zero()

    # Active but stale avoidance is a risk, not a free path.
    arb3 = CommandArbiter(command_timeout=0.5)
    arb3.set_avoidance(VelocityCommand(0.5), active=True, now=0.0)
    cmd, source = arb3.select(1.0)
    assert source == "avoidance_stale_hover" and cmd.is_zero()


def test_emergency_stop_latches():
    arb = CommandArbiter()
    arb.set_manual(VelocityCommand(2.0), now=0.0)
    arb.emergency_stop()
    cmd, source = arb.select(0.0)
    assert source == "emergency_stop" and cmd.is_zero()
    arb.clear_emergency_stop()
    cmd, source = arb.select(0.0)
    assert source == "manual"


def test_supervisor_telemetry_loss_hover():
    sup = SafetySupervisor(FlightLimits(heartbeat_timeout=1.0))
    sup.update_telemetry(0.0)
    sup.arbiter.set_manual(VelocityCommand(2.0), now=0.0)
    status = sup.evaluate(0.1, position_enu=(0, 0, 3))
    assert status.reason == "manual" and status.command.vx > 0.0
    status = sup.evaluate(2.0, position_enu=(0, 0, 3))
    assert status.reason == "telemetry_lost_hover" and status.command.is_zero()


def test_supervisor_rate_limits_and_clamps():
    sup = SafetySupervisor(FlightLimits(max_speed=1.0, max_acceleration=1.0))
    sup.update_telemetry(0.0)
    sup.arbiter.set_manual(VelocityCommand(5.0), now=0.0)
    status = sup.evaluate(0.25, position_enu=(0, 0, 3))
    # First cycle applies the command directly, clamped to max_speed.
    assert status.command.vx == 1.0
    assert status.limited is True
    status = sup.evaluate(0.5, position_enu=(0, 0, 3))
    assert status.command.vx <= 1.0
