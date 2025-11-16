"""
Traffic signal optimization using priority queues and greedy scheduling
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import heapq
from collections import defaultdict


class SignalState(str, Enum):
    """Traffic signal states"""
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"


class Direction(str, Enum):
    """Traffic flow directions"""
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    NORTH_SOUTH = "north_south"
    EAST_WEST = "east_west"


@dataclass(order=True)
class SignalRequest:
    """
    Priority queue item for signal scheduling

    Uses priority for ordering in heap
    """
    priority: int  # Higher = more urgent
    timestamp: datetime = field(compare=False)
    signal_id: int = field(compare=False)
    direction: Direction = field(compare=False)
    duration: int = field(compare=False)  # seconds
    reason: str = field(compare=False)  # emergency, congestion, schedule
    vehicle_count: int = field(compare=False, default=0)
    metadata: Dict = field(compare=False, default_factory=dict)

    def __post_init__(self):
        # Negate priority for min-heap to work as max-heap
        self.priority = -self.priority


@dataclass
class SignalTiming:
    """Traffic signal timing configuration"""
    signal_id: int
    current_state: SignalState
    current_direction: Direction
    green_duration: int = 30  # seconds
    yellow_duration: int = 5  # seconds
    red_duration: int = 55  # seconds
    last_state_change: Optional[datetime] = None
    adaptive_mode: bool = True


class SignalOptimizer:
    """
    Traffic signal optimization using priority queue scheduling

    Key algorithms:
    1. Priority Queue: O(log N) insert/extract for signal scheduling
    2. Greedy Scheduling: Maximize traffic throughput
    3. Emergency Vehicle Priority: Preempt normal traffic flow
    """

    def __init__(
        self,
        default_green_duration: int = 30,
        min_green_duration: int = 10,
        max_green_duration: int = 120,
        emergency_priority: int = 1000,
    ):
        self.default_green_duration = default_green_duration
        self.min_green_duration = min_green_duration
        self.max_green_duration = max_green_duration
        self.emergency_priority = emergency_priority

        # Priority queue for signal requests (min-heap, negated priorities)
        self.request_queue: List[SignalRequest] = []

        # Current signal states
        self.signal_states: Dict[int, SignalTiming] = {}

        # Traffic queue lengths by signal and direction
        self.traffic_queues: Dict[Tuple[int, Direction], int] = defaultdict(int)

    def add_signal(self, signal_id: int, initial_direction: Direction = Direction.NORTH_SOUTH):
        """Register a traffic signal"""
        self.signal_states[signal_id] = SignalTiming(
            signal_id=signal_id,
            current_state=SignalState.GREEN,
            current_direction=initial_direction,
            last_state_change=datetime.utcnow(),
        )

    def request_green_light(
        self,
        signal_id: int,
        direction: Direction,
        vehicle_count: int = 0,
        is_emergency: bool = False,
        reason: str = "schedule",
    ):
        """
        Request green light for a direction

        Time Complexity: O(log N) for heap insertion

        Args:
            signal_id: Traffic signal ID
            direction: Direction requesting green
            vehicle_count: Number of vehicles waiting
            is_emergency: Emergency vehicle present
            reason: Reason for request
        """
        # Calculate priority
        priority = self._calculate_priority(
            signal_id=signal_id,
            direction=direction,
            vehicle_count=vehicle_count,
            is_emergency=is_emergency,
        )

        # Calculate duration based on traffic
        duration = self._calculate_duration(vehicle_count, is_emergency)

        # Create request
        request = SignalRequest(
            priority=priority,
            timestamp=datetime.utcnow(),
            signal_id=signal_id,
            direction=direction,
            duration=duration,
            reason="emergency" if is_emergency else reason,
            vehicle_count=vehicle_count,
        )

        # Add to priority queue
        heapq.heappush(self.request_queue, request)

    def process_next_request(self) -> Optional[SignalRequest]:
        """
        Process next highest priority signal request

        Time Complexity: O(log N) for heap extraction

        Returns:
            SignalRequest if queue not empty, None otherwise
        """
        if not self.request_queue:
            return None

        # Extract highest priority request
        request = heapq.heappop(self.request_queue)

        # Apply signal change
        self._apply_signal_change(request)

        return request

    def optimize_signal_timing(
        self,
        signal_id: int,
        traffic_data: Dict[Direction, int],
    ) -> Dict[Direction, int]:
        """
        Optimize signal timing based on current traffic conditions

        Uses greedy algorithm to maximize throughput

        Args:
            signal_id: Signal to optimize
            traffic_data: Dictionary of direction -> vehicle count

        Returns:
            Dictionary of direction -> recommended green duration
        """
        total_vehicles = sum(traffic_data.values())

        if total_vehicles == 0:
            # Use default timing if no traffic
            return {
                direction: self.default_green_duration
                for direction in traffic_data.keys()
            }

        # Calculate cycle length (total time for all directions)
        total_cycle = self.default_green_duration * len(traffic_data)

        # Allocate green time proportional to traffic volume
        timing = {}
        for direction, count in traffic_data.items():
            proportion = count / total_vehicles
            duration = int(proportion * total_cycle)

            # Clamp to min/max bounds
            duration = max(self.min_green_duration, min(self.max_green_duration, duration))
            timing[direction] = duration

        return timing

    def schedule_emergency_vehicle(
        self,
        path_signal_ids: List[int],
        direction_per_signal: List[Direction],
        estimated_arrival_times: List[datetime],
    ):
        """
        Schedule green lights along emergency vehicle path

        Time Complexity: O(K log N) where K = path length

        Args:
            path_signal_ids: List of signal IDs along path
            direction_per_signal: Direction at each signal
            estimated_arrival_times: Expected arrival time at each signal
        """
        for signal_id, direction, eta in zip(
            path_signal_ids, direction_per_signal, estimated_arrival_times
        ):
            # High priority request for emergency
            self.request_green_light(
                signal_id=signal_id,
                direction=direction,
                vehicle_count=1,
                is_emergency=True,
                reason="emergency_vehicle",
            )

    def update_traffic_queue(self, signal_id: int, direction: Direction, vehicle_count: int):
        """Update vehicle count in traffic queue"""
        self.traffic_queues[(signal_id, direction)] = vehicle_count

    def get_current_state(self, signal_id: int) -> Optional[SignalTiming]:
        """Get current state of a traffic signal"""
        return self.signal_states.get(signal_id)

    def get_queue_stats(self) -> Dict:
        """Get statistics about request queue"""
        return {
            "queue_length": len(self.request_queue),
            "total_signals": len(self.signal_states),
            "pending_requests": len(self.request_queue),
        }

    def _calculate_priority(
        self,
        signal_id: int,
        direction: Direction,
        vehicle_count: int,
        is_emergency: bool,
    ) -> int:
        """
        Calculate priority for signal request

        Priority factors:
        1. Emergency vehicles: Highest priority
        2. Vehicle count: More vehicles = higher priority
        3. Wait time: Longer wait = higher priority
        """
        if is_emergency:
            return self.emergency_priority

        # Base priority from vehicle count
        priority = vehicle_count

        # Check wait time
        signal_state = self.signal_states.get(signal_id)
        if signal_state and signal_state.last_state_change:
            wait_time = (datetime.utcnow() - signal_state.last_state_change).total_seconds()

            # Add bonus for long waits (prevents starvation)
            if wait_time > 60:  # More than 1 minute
                priority += int(wait_time / 10)

        return priority

    def _calculate_duration(self, vehicle_count: int, is_emergency: bool) -> int:
        """
        Calculate green light duration based on traffic

        Uses empirical formula: duration = base + (vehicles * factor)
        """
        if is_emergency:
            return self.min_green_duration  # Quick green for emergency

        # Base duration + additional time per vehicle
        duration = self.default_green_duration + (vehicle_count * 2)

        # Clamp to bounds
        return max(self.min_green_duration, min(self.max_green_duration, duration))

    def _apply_signal_change(self, request: SignalRequest):
        """Apply signal state change"""
        if request.signal_id not in self.signal_states:
            return

        signal = self.signal_states[request.signal_id]

        # Update signal state
        signal.current_state = SignalState.GREEN
        signal.current_direction = request.direction
        signal.green_duration = request.duration
        signal.last_state_change = datetime.utcnow()

    def clear_queue(self):
        """Clear all pending requests"""
        self.request_queue.clear()

    def get_next_signal_changes(self, limit: int = 10) -> List[SignalRequest]:
        """
        Peek at next signal changes without removing from queue

        Args:
            limit: Maximum number of requests to return

        Returns:
            List of upcoming signal requests
        """
        return heapq.nsmallest(limit, self.request_queue)


class AdaptiveSignalController:
    """
    Adaptive traffic signal control using real-time data

    Implements Webster's method for optimal cycle length
    """

    def __init__(self, optimizer: SignalOptimizer):
        self.optimizer = optimizer
        self.history: List[Dict] = []

    def calculate_optimal_cycle(
        self,
        traffic_flows: Dict[Direction, float],  # vehicles per hour
        saturation_flows: Dict[Direction, float],  # max vehicles per hour
    ) -> int:
        """
        Calculate optimal cycle length using Webster's method

        Formula: C = (1.5L + 5) / (1 - Y)
        where:
            L = total lost time per cycle
            Y = sum of critical flow ratios

        Returns:
            Optimal cycle length in seconds
        """
        # Lost time per phase (startup + clearance)
        lost_time = 5  # seconds per phase

        # Calculate flow ratios (y = flow / saturation_flow)
        flow_ratios = {}
        for direction in traffic_flows:
            if saturation_flows.get(direction, 0) > 0:
                flow_ratios[direction] = traffic_flows[direction] / saturation_flows[direction]
            else:
                flow_ratios[direction] = 0

        # Sum of critical flow ratios
        Y = sum(flow_ratios.values())

        # Prevent division by zero
        if Y >= 1.0:
            Y = 0.9  # Cap at 90% saturation

        # Total lost time
        L = lost_time * len(traffic_flows)

        # Webster's formula
        optimal_cycle = (1.5 * L + 5) / (1 - Y)

        # Clamp to reasonable bounds (30-180 seconds)
        return int(max(30, min(180, optimal_cycle)))

    def update_signal_plan(
        self,
        signal_id: int,
        traffic_data: Dict[Direction, int],
    ):
        """
        Update signal plan based on current traffic

        This runs periodically (e.g., every 5 minutes) in production
        """
        # Get optimized timing
        timing = self.optimizer.optimize_signal_timing(signal_id, traffic_data)

        # Update signal configuration
        if signal_id in self.optimizer.signal_states:
            signal = self.optimizer.signal_states[signal_id]
            if timing:
                avg_duration = sum(timing.values()) // len(timing)
                signal.green_duration = avg_duration

        # Log for analysis
        self.history.append({
            "signal_id": signal_id,
            "timestamp": datetime.utcnow(),
            "traffic_data": traffic_data,
            "timing": timing,
        })
