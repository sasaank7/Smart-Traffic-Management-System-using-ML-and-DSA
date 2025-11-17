"""
Routing algorithms: Dijkstra and A* for shortest path computation
"""
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import heapq
import time
from app.services.dsa.graph import TrafficGraph, Node, Edge


class RouteAlgorithm(str, Enum):
    """Available routing algorithms"""
    DIJKSTRA = "dijkstra"
    A_STAR = "a_star"


@dataclass
class RouteResult:
    """Result of route computation"""
    path: List[int]  # List of node IDs
    segments: List[int]  # List of segment IDs
    total_distance: float  # meters
    total_time: float  # seconds
    algorithm: str
    computation_time: float  # seconds
    nodes_explored: int
    is_emergency: bool = False
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RoutingService:
    """
    Traffic routing service using graph algorithms

    Implements:
    - Dijkstra's Algorithm: O((V + E) log V) with min-heap
    - A* Algorithm: O((V + E) log V) with heuristic optimization
    """

    def __init__(self, graph: TrafficGraph):
        self.graph = graph

    def compute_route(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        algorithm: RouteAlgorithm = RouteAlgorithm.A_STAR,
        is_emergency: bool = False,
    ) -> Optional[RouteResult]:
        """
        Compute optimal route between two coordinates

        Args:
            origin_lat: Origin latitude
            origin_lon: Origin longitude
            dest_lat: Destination latitude
            dest_lon: Destination longitude
            algorithm: Routing algorithm to use
            is_emergency: Emergency vehicle flag (modifies edge weights)

        Returns:
            RouteResult with path and metrics, or None if no path exists
        """
        # Find nearest nodes to origin and destination
        start_node = self.graph.find_nearest_node(origin_lat, origin_lon)
        end_node = self.graph.find_nearest_node(dest_lat, dest_lon)

        if start_node is None or end_node is None:
            return None

        # Select algorithm
        if algorithm == RouteAlgorithm.DIJKSTRA:
            return self.dijkstra(start_node, end_node, is_emergency)
        elif algorithm == RouteAlgorithm.A_STAR:
            return self.a_star(start_node, end_node, is_emergency)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def dijkstra(
        self,
        start_node: int,
        end_node: int,
        is_emergency: bool = False,
    ) -> Optional[RouteResult]:
        """
        Dijkstra's shortest path algorithm

        Time Complexity: O((V + E) log V) using min-heap
        Space Complexity: O(V)

        Algorithm:
        1. Initialize distances to infinity except start (0)
        2. Use min-heap to process nodes in order of distance
        3. For each node, relax edges to neighbors
        4. Track predecessors for path reconstruction
        """
        start_time = time.time()
        nodes_explored = 0

        # Initialize data structures
        distances = {node_id: float('inf') for node_id in self.graph.nodes}
        distances[start_node] = 0
        predecessors = {}
        visited = set()

        # Min-heap: (distance, node_id)
        heap = [(0, start_node)]

        while heap:
            current_dist, current_node = heapq.heappop(heap)
            nodes_explored += 1

            # Skip if already visited
            if current_node in visited:
                continue

            visited.add(current_node)

            # Found destination
            if current_node == end_node:
                break

            # Explore neighbors
            for neighbor_id, edge in self.graph.get_neighbors(current_node):
                if neighbor_id in visited:
                    continue

                # Calculate edge weight
                weight = self._get_edge_weight(edge, is_emergency)
                new_distance = current_dist + weight

                # Relax edge if shorter path found
                if new_distance < distances[neighbor_id]:
                    distances[neighbor_id] = new_distance
                    predecessors[neighbor_id] = (current_node, edge.segment_id)
                    heapq.heappush(heap, (new_distance, neighbor_id))

        # Check if path exists
        if end_node not in predecessors and start_node != end_node:
            return None

        # Reconstruct path
        path, segments = self._reconstruct_path(start_node, end_node, predecessors)
        total_distance = self._calculate_total_distance(segments)

        computation_time = time.time() - start_time

        return RouteResult(
            path=path,
            segments=segments,
            total_distance=total_distance,
            total_time=distances[end_node],
            algorithm=RouteAlgorithm.DIJKSTRA,
            computation_time=computation_time,
            nodes_explored=nodes_explored,
            is_emergency=is_emergency,
        )

    def a_star(
        self,
        start_node: int,
        end_node: int,
        is_emergency: bool = False,
    ) -> Optional[RouteResult]:
        """
        A* shortest path algorithm with haversine heuristic

        Time Complexity: O((V + E) log V) - typically faster than Dijkstra
        Space Complexity: O(V)

        Algorithm:
        1. Similar to Dijkstra but uses f(n) = g(n) + h(n)
        2. g(n) = actual distance from start
        3. h(n) = heuristic (haversine distance to goal)
        4. Guarantees optimal path if heuristic is admissible
        """
        start_time = time.time()
        nodes_explored = 0

        # Initialize data structures
        g_scores = {node_id: float('inf') for node_id in self.graph.nodes}
        g_scores[start_node] = 0

        f_scores = {node_id: float('inf') for node_id in self.graph.nodes}
        f_scores[start_node] = self.graph.haversine_distance(start_node, end_node)

        predecessors = {}
        visited = set()

        # Min-heap: (f_score, g_score, node_id)
        heap = [(f_scores[start_node], 0, start_node)]

        while heap:
            current_f, current_g, current_node = heapq.heappop(heap)
            nodes_explored += 1

            # Skip if already visited
            if current_node in visited:
                continue

            visited.add(current_node)

            # Found destination
            if current_node == end_node:
                break

            # Explore neighbors
            for neighbor_id, edge in self.graph.get_neighbors(current_node):
                if neighbor_id in visited:
                    continue

                # Calculate tentative g score
                weight = self._get_edge_weight(edge, is_emergency)
                tentative_g = g_scores[current_node] + weight

                # Update if better path found
                if tentative_g < g_scores[neighbor_id]:
                    g_scores[neighbor_id] = tentative_g
                    h_score = self._heuristic(neighbor_id, end_node, is_emergency)
                    f_scores[neighbor_id] = tentative_g + h_score
                    predecessors[neighbor_id] = (current_node, edge.segment_id)
                    heapq.heappush(heap, (f_scores[neighbor_id], tentative_g, neighbor_id))

        # Check if path exists
        if end_node not in predecessors and start_node != end_node:
            return None

        # Reconstruct path
        path, segments = self._reconstruct_path(start_node, end_node, predecessors)
        total_distance = self._calculate_total_distance(segments)

        computation_time = time.time() - start_time

        return RouteResult(
            path=path,
            segments=segments,
            total_distance=total_distance,
            total_time=g_scores[end_node],
            algorithm=RouteAlgorithm.A_STAR,
            computation_time=computation_time,
            nodes_explored=nodes_explored,
            is_emergency=is_emergency,
        )

    def _get_edge_weight(self, edge: Edge, is_emergency: bool) -> float:
        """
        Calculate edge weight for routing

        For emergency vehicles, reduce weight significantly
        """
        weight = edge.current_travel_time

        if is_emergency:
            # Emergency vehicles get priority - reduce effective travel time
            weight *= 0.2  # 80% reduction

        return weight

    def _heuristic(self, node_id: int, goal_id: int, is_emergency: bool) -> float:
        """
        A* heuristic function (haversine distance / max speed)

        Must be admissible (never overestimate) for optimal solution
        """
        distance = self.graph.haversine_distance(node_id, goal_id)

        # Assume maximum highway speed for time estimate
        max_speed_mps = 120 * 1000 / 3600  # 120 km/h in m/s
        if is_emergency:
            max_speed_mps *= 1.5  # Emergency vehicles can go faster

        return distance / max_speed_mps

    def _reconstruct_path(
        self,
        start_node: int,
        end_node: int,
        predecessors: Dict[int, Tuple[int, int]],
    ) -> Tuple[List[int], List[int]]:
        """
        Reconstruct path from predecessors dictionary

        Returns:
            (node_path, segment_path)
        """
        if start_node == end_node:
            return [start_node], []

        path = []
        segments = []
        current = end_node

        while current in predecessors:
            path.append(current)
            pred_node, segment_id = predecessors[current]
            segments.append(segment_id)
            current = pred_node

        path.append(start_node)
        path.reverse()
        segments.reverse()

        return path, segments

    def _calculate_total_distance(self, segment_ids: List[int]) -> float:
        """Calculate total distance of route in meters"""
        total = 0.0
        for segment_id in segment_ids:
            if segment_id in self.graph._edge_by_segment:
                total += self.graph._edge_by_segment[segment_id].length
        return total

    def get_alternative_routes(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        num_alternatives: int = 3,
    ) -> List[RouteResult]:
        """
        Generate alternative routes by penalizing main route segments

        Uses iterative penalty method
        """
        routes = []

        # Get main route
        main_route = self.compute_route(origin_lat, origin_lon, dest_lat, dest_lon)
        if main_route:
            routes.append(main_route)

        # Generate alternatives by penalizing segments from previous routes
        penalty_factor = 1.5
        penalized_segments = set()

        for i in range(num_alternatives - 1):
            if not routes:
                break

            # Penalize segments from last route
            for segment_id in routes[-1].segments:
                penalized_segments.add(segment_id)
                if segment_id in self.graph._edge_by_segment:
                    edge = self.graph._edge_by_segment[segment_id]
                    original_factor = edge.congestion_factor
                    edge.congestion_factor *= penalty_factor

            # Compute alternative route
            alt_route = self.compute_route(origin_lat, origin_lon, dest_lat, dest_lon)

            if alt_route and alt_route.segments != routes[-1].segments:
                routes.append(alt_route)

        # Restore original congestion factors
        for segment_id in penalized_segments:
            if segment_id in self.graph._edge_by_segment:
                edge = self.graph._edge_by_segment[segment_id]
                edge.congestion_factor /= (penalty_factor ** (num_alternatives - 1))

        return routes
