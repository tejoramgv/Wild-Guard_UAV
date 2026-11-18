"""Flight-state transition tests."""

from forest_bringup.flight_state import FlightState, FlightStateMachine


def test_takeoff_sequence():
    sm = FlightStateMachine(hover_altitude=3.0)
    assert sm.state == FlightState.LANDED
    assert sm.request_takeoff() is True
    assert sm.state == FlightState.TAKEOFF_REQUESTED
    assert sm.on_odometry(1.0) == FlightState.TAKEOFF_REQUESTED
    assert sm.on_odometry(2.5) == FlightState.FLYING
    assert sm.command_allowed()


def test_hover_and_move_transitions():
    sm = FlightStateMachine(hover_altitude=3.0, state=FlightState.FLYING)
    assert sm.on_command(False) == FlightState.HOVER
    assert sm.on_command(True) == FlightState.FLYING


def test_land_sequence():
    sm = FlightStateMachine(hover_altitude=3.0, state=FlightState.FLYING)
    assert sm.request_land() is True
    assert sm.state == FlightState.LAND_REQUESTED
    assert sm.on_odometry(2.0) == FlightState.LAND_REQUESTED
    assert sm.on_odometry(0.1) == FlightState.LANDED
    assert sm.is_landed()


def test_emergency_latch_and_clear():
    sm = FlightStateMachine(state=FlightState.FLYING)
    sm.emergency_stop()
    assert sm.state == FlightState.EMERGENCY_STOP
    assert sm.command_allowed() is False
    assert sm.clear_emergency() is True
    assert sm.state == FlightState.HOVER


def test_invalid_transitions_rejected():
    sm = FlightStateMachine()
    assert sm.request_land() is False
    assert sm.request_hover() is False
    assert sm.request_takeoff() is True
    assert sm.request_takeoff() is False
