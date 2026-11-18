"""Pure flight-state machine used by the ROS executor node."""

from dataclasses import dataclass
from enum import Enum


class FlightState(str, Enum):
    LANDED = "landed"
    TAKEOFF_REQUESTED = "takeoff_requested"
    FLYING = "flying"
    HOVER = "hover"
    LAND_REQUESTED = "land_requested"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class FlightStateMachine:
    hover_altitude: float = 3.0
    state: FlightState = FlightState.LANDED
    altitude: float = 0.0

    def request_takeoff(self):
        if self.state in (FlightState.LANDED, FlightState.HOVER):
            self.state = FlightState.TAKEOFF_REQUESTED
            return True
        return False

    def request_land(self):
        if self.state in (FlightState.FLYING, FlightState.HOVER,
                          FlightState.TAKEOFF_REQUESTED):
            self.state = FlightState.LAND_REQUESTED
            return True
        return False

    def request_hover(self):
        if self.state in (FlightState.FLYING, FlightState.HOVER):
            self.state = FlightState.HOVER
            return True
        return False

    def emergency_stop(self):
        self.state = FlightState.EMERGENCY_STOP
        return True

    def clear_emergency(self):
        if self.state == FlightState.EMERGENCY_STOP:
            self.state = FlightState.HOVER
            return True
        return False

    def on_odometry(self, altitude):
        self.altitude = altitude
        if self.state == FlightState.TAKEOFF_REQUESTED:
            if altitude >= self.hover_altitude * 0.8:
                self.state = FlightState.FLYING
        elif self.state == FlightState.LAND_REQUESTED:
            if altitude <= 0.15:
                self.state = FlightState.LANDED
        return self.state

    def on_command(self, moving):
        if self.state == FlightState.FLYING and not moving:
            self.state = FlightState.HOVER
        elif self.state == FlightState.HOVER and moving:
            self.state = FlightState.FLYING
        return self.state

    def command_allowed(self):
        return self.state in (FlightState.FLYING, FlightState.HOVER)

    def is_landed(self):
        return self.state == FlightState.LANDED
